// fuse::petsc::MffdToken/make_assembled_token -- the PETSc-backed
// fuse::LinearizationToken pair (fuse/petsc/token.hpp). Mirrors
// test/num/newton_krylov_test.cpp's own Mffd.* section: the SAME
// quadratic test problem (F_i(x) = x_i^2 - c_i, J(x) = diag(2 x_i)),
// so the two substrates' own measured `h` and Jacobian-action error are
// directly comparable (docs/dev/newton-krylov.md Sec. 10's own table).

#include "fuse/petsc/token.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "fuse/linalg/comm.hpp"
#include "fuse/linalg/dist_vector.hpp"
#include "fuse/num/newton_krylov.hpp"

using fuse::Failure;
using fuse::Result;
using fuse::linalg::DistVector;
using fuse::linalg::make_serial_exchanger;
using fuse::linalg::make_serial_reducer;
using fuse::num::default_mffd_error_rel;
using fuse::num::MffdJacobianAction;
using fuse::num::MffdOptions;
using fuse::num::PreconditionerBlock;
using fuse::num::ResidualFn;
using fuse::petsc::comm_self_f;
using fuse::petsc::kNoPointCode;
using fuse::petsc::kNoTransposeCode;
using fuse::petsc::make_assembled_token;
using fuse::petsc::make_mffd_token;
using fuse::petsc::MatrixTriplet;
using fuse::petsc::MffdToken;
using fuse::petsc::MffdTokenOptions;
using fuse::petsc::ResidualSpanFn;

namespace {

/// F_i(x) = x_i^2 - c_i -- fuse::petsc's own span-based convention.
ResidualSpanFn quadratic_span(const std::vector<double>& c) {
  return [c](std::span<const double> x, std::span<double> fx) -> Result<void> {
    for (std::size_t i = 0; i < c.size(); ++i) fx[i] = x[i] * x[i] - c[i];
    return Result<void>::success();
  };
}

/// The SAME problem, DistVector-based (fuse::num's own convention) --
/// test/num/newton_krylov_test.cpp's own `quadratic_residual`.
ResidualFn quadratic_distvec(const std::vector<double>& c) {
  return [c](DistVector& x, DistVector& fx) -> Result<void> {
    for (std::size_t i = 0; i < c.size(); ++i) fx.data()[i] = x.data()[i] * x.data()[i] - c[i];
    return Result<void>::success();
  };
}

DistVector vec_sized(std::size_t n) {
  return DistVector(n, make_serial_exchanger(), make_serial_reducer());
}

DistVector vec_from(const std::vector<double>& v) {
  DistVector out = vec_sized(v.size());
  std::copy(v.begin(), v.end(), out.data());
  return out;
}

}  // namespace

TEST(MffdToken, ApplyJacobianBeforeBasePointIsRefused) {
  auto made = make_mffd_token(quadratic_span({4.0}), 1, comm_self_f());
  ASSERT_TRUE(made.ok()) << made.error().code << ": " << made.error().message;
  auto token = std::move(made).value();
  std::vector<double> v = {1.0}, y(1, 0.0);
  Result<void> r = token->apply_jacobian(v, y);
  ASSERT_FALSE(r.ok());
  EXPECT_STREQ(r.error().code.c_str(), kNoPointCode);
}

TEST(MffdToken, TransposeAlwaysRefused) {
  auto made = make_mffd_token(quadratic_span({4.0}), 1, comm_self_f());
  ASSERT_TRUE(made.ok()) << made.error().code << ": " << made.error().message;
  auto token = std::move(made).value();
  std::vector<double> x0 = {3.0};
  ASSERT_TRUE(token->set_base_point(x0).ok());
  std::vector<double> v = {1.0}, y(1, 0.0);
  Result<void> r = token->apply_jacobian_transpose(v, y);
  ASSERT_FALSE(r.ok());
  EXPECT_STREQ(r.error().code.c_str(), kNoTransposeCode);
}

TEST(MffdToken, ApplyJacobianAgreesWithAnalyticJacobianAndRecordsHAlongsideTheNativeAction) {
  const std::vector<double> c = {4.0, 9.0};
  const std::vector<double> x0 = {3.0, 1.0};
  const std::vector<double> v0 = {1.0, 0.5};

  auto made = make_mffd_token(quadratic_span(c), 2, comm_self_f());
  ASSERT_TRUE(made.ok()) << made.error().code << ": " << made.error().message;
  auto token = std::move(made).value();
  ASSERT_TRUE(token->set_base_point(x0).ok());

  std::vector<double> y(2, 0.0);
  ASSERT_TRUE(token->apply_jacobian(v0, y).ok());

  const double analytic0 = 2.0 * x0[0] * v0[0];
  const double analytic1 = 2.0 * x0[1] * v0[1];
  const double sqrt_eps = std::sqrt(std::numeric_limits<double>::epsilon());
  EXPECT_NEAR(y[0], analytic0, 100.0 * sqrt_eps * std::max(1.0, std::fabs(analytic0)));
  EXPECT_NEAR(y[1], analytic1, 100.0 * sqrt_eps * std::max(1.0, std::fabs(analytic1)));

  // The native action's own h, at the IDENTICAL point/direction/error_rel,
  // for direct comparison -- recorded to 17 digits (round-trip precision
  // for a double).
  DistVector xv = vec_from(x0);
  DistVector fxv = vec_sized(2);
  ASSERT_TRUE(quadratic_distvec(c)(xv, fxv).ok());
  MffdOptions native_opt;
  MffdJacobianAction native(quadratic_distvec(c), native_opt);
  native.set_point(xv, fxv);
  DistVector vv = vec_from(v0);
  DistVector jv = vec_sized(2);
  ASSERT_TRUE(native.apply(vv, jv).ok());
  ASSERT_EQ(native.trace().size(), 1u);
  const double native_h = native.trace().front().h;
  const double petsc_h = token->last_h();

  std::fprintf(stderr, "[petsc.token h-comparison] native_h=%.17g petsc_h=%.17g\n", native_h,
              petsc_h);
  EXPECT_GT(petsc_h, 0.0);
  EXPECT_GT(native_h, 0.0);
}

TEST(MffdToken, ApplyPreconditionerDefaultIsIdentity) {
  auto made = make_mffd_token(quadratic_span({4.0, 9.0}), 2, comm_self_f());
  ASSERT_TRUE(made.ok());
  auto token = std::move(made).value();
  ASSERT_TRUE(token->set_base_point(std::vector<double>{3.0, 1.0}).ok());
  std::vector<double> r = {5.0, -2.0}, z(2, 0.0);
  ASSERT_TRUE(token->apply_preconditioner(r, z).ok());
  EXPECT_DOUBLE_EQ(z[0], r[0]);
  EXPECT_DOUBLE_EQ(z[1], r[1]);
}

TEST(MffdToken, BlockPreconditionerMatchesTheExactPerBlockInverse) {
  // Mirrors test/num/newton_krylov_test.cpp's own atan-residual exact
  // block preconditioner: each coordinate's own exact inverse is
  // (1 + x_i^2), applied as a 1-wide block.
  auto made = make_mffd_token(quadratic_span({4.0, 9.0, 1.0}), 3, comm_self_f());
  ASSERT_TRUE(made.ok());
  auto token = std::move(made).value();
  ASSERT_TRUE(token->set_base_point(std::vector<double>{3.0, 1.0, 2.0}).ok());

  const std::vector<double> scale = {10.0, 2.0, 5.0};  // 1 + x_i^2 analogue, arbitrary
  std::vector<PreconditionerBlock> blocks;
  for (std::size_t i = 0; i < scale.size(); ++i) {
    blocks.push_back(PreconditionerBlock{
        i, 1, [s = scale[i]](std::span<const double> r, std::span<double> z) -> Result<void> {
          z[0] = r[0] * s;
          return Result<void>::success();
        }});
  }
  ASSERT_TRUE(token->set_block_preconditioner(std::move(blocks)).ok());

  std::vector<double> r = {1.0, 2.0, 3.0}, z(3, 0.0);
  ASSERT_TRUE(token->apply_preconditioner(r, z).ok());
  EXPECT_DOUBLE_EQ(z[0], r[0] * scale[0]);
  EXPECT_DOUBLE_EQ(z[1], r[1] * scale[1]);
  EXPECT_DOUBLE_EQ(z[2], r[2] * scale[2]);
}

TEST(MffdToken, BlockPreconditionerCoverageGapIsRefused) {
  auto made = make_mffd_token(quadratic_span({4.0, 9.0}), 2, comm_self_f());
  ASSERT_TRUE(made.ok());
  auto token = std::move(made).value();
  ASSERT_TRUE(token->set_base_point(std::vector<double>{3.0, 1.0}).ok());
  // Only covers index 0 of a 2-wide vector: a gap.
  std::vector<PreconditionerBlock> blocks;
  blocks.push_back(PreconditionerBlock{
      0, 1, [](std::span<const double> r, std::span<double> z) -> Result<void> {
        z[0] = r[0];
        return Result<void>::success();
      }});
  Result<void> r = token->set_block_preconditioner(std::move(blocks));
  ASSERT_FALSE(r.ok());
  EXPECT_STREQ(r.error().code.c_str(), "E-NEWTON-PRECOND-COVERAGE");
}

TEST(MffdToken, RowsColsAndSignature) {
  auto made = make_mffd_token(quadratic_span({4.0, 9.0, 1.0}), 3, comm_self_f());
  ASSERT_TRUE(made.ok());
  auto token = std::move(made).value();
  EXPECT_EQ(token->rows(), 3u);
  EXPECT_EQ(token->cols(), 3u);
  const std::uint64_t sig_before = token->signature();
  ASSERT_TRUE(token->set_base_point(std::vector<double>{1.0, 1.0, 1.0}).ok());
  const std::uint64_t sig_after_first = token->signature();
  EXPECT_NE(sig_before, sig_after_first);
  ASSERT_TRUE(token->set_base_point(std::vector<double>{2.0, 2.0, 2.0}).ok());
  const std::uint64_t sig_after_second = token->signature();
  EXPECT_NE(sig_after_first, sig_after_second);
}

TEST(MffdToken, ShapeMismatchIsRefused) {
  auto made = make_mffd_token(quadratic_span({4.0, 9.0}), 2, comm_self_f());
  ASSERT_TRUE(made.ok());
  auto token = std::move(made).value();
  std::vector<double> bad_x = {1.0, 2.0, 3.0};
  Result<void> r = token->set_base_point(bad_x);
  ASSERT_FALSE(r.ok());
  EXPECT_STREQ(r.error().code.c_str(), "E-PETSC-MFFD-SHAPE");
}

// ---------------------------------------------------------------------------
// The assembled token: a real MATAIJ, both apply_jacobian and
// apply_jacobian_transpose genuine.
// ---------------------------------------------------------------------------

TEST(AssembledToken, ApplyJacobianAndTransposeMatchAHandComputedMatrix) {
  // A = [[2, 0, 1], [0, 3, 0]] (2x3, deliberately non-square and
  // non-symmetric so Mv and M^T w exercise genuinely different code paths).
  std::vector<MatrixTriplet> triplets = {
      {0, 0, 2.0}, {0, 2, 1.0}, {1, 1, 3.0},
  };
  auto made = make_assembled_token(2, 3, triplets, comm_self_f());
  ASSERT_TRUE(made.ok()) << made.error().code << ": " << made.error().message;
  auto token = std::move(made).value();
  EXPECT_EQ(token->rows(), 2u);
  EXPECT_EQ(token->cols(), 3u);

  std::vector<double> x = {1.0, 2.0, 3.0};
  std::vector<double> y(2, 0.0);
  ASSERT_TRUE(token->apply_jacobian(x, y).ok());
  // A x = [2*1 + 1*3, 3*2] = [5, 6]
  EXPECT_DOUBLE_EQ(y[0], 5.0);
  EXPECT_DOUBLE_EQ(y[1], 6.0);

  std::vector<double> w = {2.0, -1.0};
  std::vector<double> z(3, 0.0);
  ASSERT_TRUE(token->apply_jacobian_transpose(w, z).ok());
  // A^T w = [2*2, 3*-1, 1*2] = [4, -3, 2]
  EXPECT_DOUBLE_EQ(z[0], 4.0);
  EXPECT_DOUBLE_EQ(z[1], -3.0);
  EXPECT_DOUBLE_EQ(z[2], 2.0);
}

TEST(AssembledToken, DuplicateTripletsAccumulate) {
  std::vector<MatrixTriplet> triplets = {{0, 0, 2.0}, {0, 0, 3.0}};  // A[0][0] = 5
  auto made = make_assembled_token(1, 1, triplets, comm_self_f());
  ASSERT_TRUE(made.ok());
  auto token = std::move(made).value();
  std::vector<double> x = {1.0}, y(1, 0.0);
  ASSERT_TRUE(token->apply_jacobian(x, y).ok());
  EXPECT_DOUBLE_EQ(y[0], 5.0);
}

TEST(AssembledToken, ApplyPreconditionerDefaultIsIdentityOnASquareOperator) {
  std::vector<MatrixTriplet> triplets = {{0, 0, 7.0}, {1, 1, -3.0}};
  auto made = make_assembled_token(2, 2, triplets, comm_self_f());
  ASSERT_TRUE(made.ok());
  auto token = std::move(made).value();
  std::vector<double> r = {4.0, -5.0}, z(2, 0.0);
  ASSERT_TRUE(token->apply_preconditioner(r, z).ok());
  EXPECT_DOUBLE_EQ(z[0], r[0]);
  EXPECT_DOUBLE_EQ(z[1], r[1]);
}

TEST(AssembledToken, ApplyPreconditionerOnANonSquareOperatorIsRefused) {
  std::vector<MatrixTriplet> triplets = {{0, 0, 2.0}, {0, 2, 1.0}, {1, 1, 3.0}};
  auto made = make_assembled_token(2, 3, triplets, comm_self_f());
  ASSERT_TRUE(made.ok());
  auto token = std::move(made).value();
  std::vector<double> r = {1.0, 1.0}, z(3, 0.0);
  Result<void> res = token->apply_preconditioner(r, z);
  ASSERT_FALSE(res.ok());
  EXPECT_STREQ(res.error().code.c_str(), "E-PETSC-PRECOND-SHAPE");
}

TEST(AssembledToken, ShapeMismatchIsRefused) {
  std::vector<MatrixTriplet> triplets = {{0, 0, 1.0}};
  auto made = make_assembled_token(1, 1, triplets, comm_self_f());
  ASSERT_TRUE(made.ok());
  auto token = std::move(made).value();
  std::vector<double> bad_x = {1.0, 2.0};
  std::vector<double> y(1, 0.0);
  Result<void> r = token->apply_jacobian(bad_x, y);
  ASSERT_FALSE(r.ok());
  EXPECT_STREQ(r.error().code.c_str(), "E-PETSC-ASSEMBLED-SHAPE");
}
