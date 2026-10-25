// The derivative gate (fuse/linalg/derivative_gate.hpp) run on the
// PETSc-backed tokens (fuse/petsc/token.hpp). Tier-0 row
// petsc.derivative_gate.
//
// Three checks, mirroring test/linalg/derivative_gate_test.cpp's own
// structure:
//   1. adjoint_gate on the ASSEMBLED token (MatMult vs MatMultTranspose --
//      the MFFD token has no transpose to test, fuse/petsc/token.hpp's
//      own kNoTransposeCode).
//   2. taylor_remainder_order on the MFFD token's action, over the SAME
//      atan and quadratic problems test/num/newton_krylov_test.cpp uses.
//   3. matrix_free_agreement between the MFFD token's action and
//      fuse::num::MffdJacobianAction's, on the SAME F/x/a -- both use
//      Walker-Pernice, but PETSc's own h and the native h can differ (the
//      two are read independently and compared to a bound derived from
//      both, never loosened to force agreement).

#include "fuse/linalg/derivative_gate.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include <gtest/gtest.h>

#include "fuse/linalg/comm.hpp"
#include "fuse/linalg/dist_vector.hpp"
#include "fuse/num/newton_krylov.hpp"
#include "fuse/petsc/token.hpp"

using fuse::Result;
using fuse::linalg::adjoint_gate;
using fuse::linalg::AdjointGateResult;
using fuse::linalg::DistVector;
using fuse::linalg::estimate_operator_norm;
using fuse::linalg::HostApplyFn;
using fuse::linalg::make_serial_exchanger;
using fuse::linalg::make_serial_reducer;
using fuse::linalg::MatrixFreeAgreementResult;
using fuse::linalg::matrix_free_agreement;
using fuse::linalg::taylor_remainder_order;
using fuse::linalg::TaylorRemainderResult;
using fuse::num::default_mffd_error_rel;
using fuse::num::MffdJacobianAction;
using fuse::num::MffdOptions;
using fuse::num::ResidualFn;
using fuse::petsc::comm_self_f;
using fuse::petsc::make_assembled_token;
using fuse::petsc::make_mffd_token;
using fuse::petsc::MatrixTriplet;
using fuse::petsc::MffdTokenOptions;
using fuse::petsc::ResidualSpanFn;

namespace {

DistVector vec_sized(std::size_t n) {
  return DistVector(n, make_serial_exchanger(), make_serial_reducer());
}

DistVector vec_from(const std::vector<double>& v) {
  DistVector out = vec_sized(v.size());
  std::copy(v.begin(), v.end(), out.data());
  return out;
}

ResidualFn quadratic_distvec(const std::vector<double>& c) {
  return [c](DistVector& x, DistVector& fx) -> Result<void> {
    for (std::size_t i = 0; i < c.size(); ++i) fx.data()[i] = x.data()[i] * x.data()[i] - c[i];
    return Result<void>::success();
  };
}

ResidualSpanFn quadratic_span(const std::vector<double>& c) {
  return [c](std::span<const double> x, std::span<double> fx) -> Result<void> {
    for (std::size_t i = 0; i < c.size(); ++i) fx[i] = x[i] * x[i] - c[i];
    return Result<void>::success();
  };
}

ResidualSpanFn atan_span(const std::vector<double>& c) {
  return [c](std::span<const double> x, std::span<double> fx) -> Result<void> {
    for (std::size_t i = 0; i < c.size(); ++i) fx[i] = std::atan(x[i]) - std::atan(c[i]);
    return Result<void>::success();
  };
}

}  // namespace

// ---------------------------------------------------------------------------
// 1. Adjoint identity on the assembled token.
// ---------------------------------------------------------------------------

TEST(PetscDerivativeGate, AdjointIdentityHoldsOnTheAssembledToken) {
  // A random-looking, deliberately non-symmetric 4x4 matrix -- the same
  // shape test/linalg/derivative_gate_test.cpp's own DistSparseMatrix case
  // uses, small enough to hand-verify the triplet list.
  std::vector<MatrixTriplet> triplets = {
      {0, 0, 4.0}, {0, 1, -1.0}, {0, 3, 0.5},  {1, 1, 3.0},
      {1, 2, 1.5}, {2, 0, 0.2},  {2, 2, 5.0},  {2, 3, -0.7},
      {3, 1, 0.3}, {3, 3, 2.0},
  };
  constexpr std::size_t n = 4;
  auto made = make_assembled_token(n, n, triplets, comm_self_f());
  ASSERT_TRUE(made.ok()) << made.error().code << ": " << made.error().message;
  auto token = std::move(made).value();

  HostApplyFn apply = [&token](std::span<const double> x, std::span<double> y) {
    return token->apply_jacobian(x, y);
  };
  HostApplyFn apply_t = [&token](std::span<const double> x, std::span<double> y) {
    return token->apply_jacobian_transpose(x, y);
  };

  Result<double> norm_est = estimate_operator_norm(n, n, apply, apply_t);
  ASSERT_TRUE(norm_est.ok()) << norm_est.error().code;

  const std::vector<double> x = {1.0, -2.0, 0.5, 3.0};
  const std::vector<double> y = {0.3, 1.0, -1.5, 2.0};
  Result<AdjointGateResult> gate =
      adjoint_gate(n, n, apply, apply_t, x, y, norm_est.value(), fuse::linalg::kAdjointGateConstant,
                  "petsc-assembled-4x4");
  ASSERT_TRUE(gate.ok()) << gate.error().code << ": " << gate.error().message;
  std::fprintf(stderr, "[petsc.derivative_gate adjoint] defect=%.17g bound=%.17g ratio=%.6g\n",
              gate.value().defect, gate.value().bound,
              gate.value().bound > 0.0 ? gate.value().defect / gate.value().bound : -1.0);
  EXPECT_TRUE(gate.value().passed) << "defect=" << gate.value().defect
                                   << " bound=" << gate.value().bound;
}

TEST(PetscDerivativeGate, ADeliberatelyWrongTransposeIsDetected) {
  // Same matrix as above, but the adjoint check is run against the
  // WRONG transpose (apply used on both sides) -- the control case
  // proving the gate actually distinguishes a real transpose bug from
  // round-off, mirroring test/linalg/derivative_gate_test.cpp's own
  // ADeliberatelyWrongAdjointIsDetected.
  std::vector<MatrixTriplet> triplets = {
      {0, 0, 4.0}, {0, 1, -1.0}, {1, 0, 2.0}, {1, 1, 3.0},
  };
  constexpr std::size_t n = 2;
  auto made = make_assembled_token(n, n, triplets, comm_self_f());
  ASSERT_TRUE(made.ok());
  auto token = std::move(made).value();

  HostApplyFn apply = [&token](std::span<const double> x, std::span<double> y) {
    return token->apply_jacobian(x, y);
  };
  HostApplyFn wrong_transpose = [&token](std::span<const double> x, std::span<double> y) {
    return token->apply_jacobian(x, y);  // deliberately NOT the transpose
  };

  Result<double> norm_est = estimate_operator_norm(n, n, apply, wrong_transpose);
  ASSERT_TRUE(norm_est.ok());
  // x and y must DIFFER: for x == y the identity <Ax, y> = <x, A^T y> reads
  // x^T A x on both sides and holds for every A, transpose or not -- using
  // (1,1) for both would pass with the wrong transpose, which is the gate
  // being blind, not the defect being absent. With these vectors <Ax, y> = -7 and <x, Ay> = 2.
  const std::vector<double> x = {1.0, 1.0};
  const std::vector<double> y = {1.0, -2.0};
  Result<AdjointGateResult> gate = adjoint_gate(n, n, apply, wrong_transpose, x, y, norm_est.value());
  ASSERT_TRUE(gate.ok());
  EXPECT_FALSE(gate.value().passed) << "a deliberately wrong transpose must fail the gate";
}

// ---------------------------------------------------------------------------
// 2. Taylor-remainder order on the MFFD token's action.
// ---------------------------------------------------------------------------

TEST(PetscDerivativeGate, TaylorRemainderOrderOnTheQuadraticProblem) {
  const std::vector<double> c = {4.0, 9.0};
  const std::vector<double> x0 = {3.0, 1.0};
  const std::vector<double> v0 = {1.0, 0.5};

  auto made = make_mffd_token(quadratic_span(c), 2, comm_self_f());
  ASSERT_TRUE(made.ok());
  auto token = std::move(made).value();
  ASSERT_TRUE(token->set_base_point(x0).ok());

  auto F = [&](const std::vector<double>& x) {
    std::vector<double> out(c.size());
    quadratic_span(c)(x, out);
    return out;
  };
  auto Jv = [&](const std::vector<double>&, const std::vector<double>& v) {
    std::vector<double> out(v.size());
    // Re-linearise at x0 every call (the token's own base point is fixed
    // for this test): apply_jacobian IS the claimed Jacobian action.
    token->apply_jacobian(v, out);
    return out;
  };

  const TaylorRemainderResult res = taylor_remainder_order(F, Jv, x0, v0, /*h0=*/1e-1,
                                                           /*ndecades=*/6);
  ASSERT_FALSE(res.at_floor_everywhere);
  std::fprintf(stderr, "[petsc.derivative_gate taylor quadratic] order=%.6g floor=%.6g\n",
              res.order, res.floor);
  EXPECT_NEAR(res.order, 2.0, 0.3) << "a quadratic residual's Taylor remainder must shrink at "
                                      "order 2 above the round-off floor";
}

TEST(PetscDerivativeGate, TaylorRemainderOrderOnTheAtanProblem) {
  const std::vector<double> c = {0.0};
  const std::vector<double> x0 = {1.0};
  const std::vector<double> v0 = {1.0};

  auto made = make_mffd_token(atan_span(c), 1, comm_self_f());
  ASSERT_TRUE(made.ok());
  auto token = std::move(made).value();
  ASSERT_TRUE(token->set_base_point(x0).ok());

  auto F = [&](const std::vector<double>& x) {
    std::vector<double> out(c.size());
    atan_span(c)(x, out);
    return out;
  };
  auto Jv = [&](const std::vector<double>&, const std::vector<double>& v) {
    std::vector<double> out(v.size());
    token->apply_jacobian(v, out);
    return out;
  };

  const TaylorRemainderResult res = taylor_remainder_order(F, Jv, x0, v0, /*h0=*/1e-1,
                                                           /*ndecades=*/6);
  std::fprintf(stderr, "[petsc.derivative_gate taylor atan] order=%.6g floor=%.6g at_floor=%d\n",
              res.order, res.floor, res.at_floor_everywhere ? 1 : 0);
  if (!res.at_floor_everywhere) {
    EXPECT_NEAR(res.order, 2.0, 0.3);
  }
}

// ---------------------------------------------------------------------------
// 3. Agreement between the PETSc MFFD action and the native MFFD action.
// ---------------------------------------------------------------------------

TEST(PetscDerivativeGate, MffdAgreesWithTheNativeActionToABoundDerivedFromBothHValues) {
  const std::vector<double> c = {4.0, 9.0};
  const std::vector<double> x0 = {3.0, 1.0};

  // The native action, at this project's own default error_rel.
  DistVector xv = vec_from(x0);
  DistVector fxv = vec_sized(2);
  ASSERT_TRUE(quadratic_distvec(c)(xv, fxv).ok());
  MffdOptions native_opt;
  MffdJacobianAction native(quadratic_distvec(c), native_opt);
  native.set_point(xv, fxv);

  // The PETSc token, at the SAME x0 and the SAME error_rel (MffdTokenOptions'
  // own default is fuse::num::default_mffd_error_rel(), fuse/petsc/token.hpp).
  auto made = make_mffd_token(quadratic_span(c), 2, comm_self_f());
  ASSERT_TRUE(made.ok());
  auto token = std::move(made).value();
  ASSERT_TRUE(token->set_base_point(x0).ok());

  HostApplyFn petsc_apply = [&token](std::span<const double> v, std::span<double> y) {
    return token->apply_jacobian(v, y);
  };
  HostApplyFn native_apply = [&native](std::span<const double> v, std::span<double> y) {
    DistVector vv = vec_sized(v.size());
    std::copy(v.begin(), v.end(), vv.data());
    DistVector jv = vec_sized(y.size());
    Result<void> r = native.apply(vv, jv);
    if (!r.ok()) return r;
    std::copy(jv.data(), jv.data() + y.size(), y.begin());
    return Result<void>::success();
  };

  // Prime both actions' own `h` (one apply() each) before deriving the
  // bound from them -- the bound is a function of the TWO h values
  // actually used, not of error_rel alone (PETSc's own ||a|| handling can
  // differ from the native formula's at the same inputs).
  {
    std::vector<double> probe_v = {1.0, 0.5}, probe_y(2, 0.0);
    ASSERT_TRUE(petsc_apply(probe_v, probe_y).ok());
    DistVector probe_vv = vec_from(probe_v);
    DistVector probe_jv = vec_sized(2);
    ASSERT_TRUE(native.apply(probe_vv, probe_jv).ok());
  }
  const double native_h = native.trace().back().h;
  const double petsc_h = token->last_h();
  std::fprintf(stderr, "[petsc.derivative_gate agreement] native_h=%.17g petsc_h=%.17g\n",
              native_h, petsc_h);

  Result<double> norm_est = estimate_operator_norm(2, 2, petsc_apply, petsc_apply);
  ASSERT_TRUE(norm_est.ok());

  // Both actions difference the SAME quadratic F by one-sided finite
  // difference; each one's own error against the TRUE Jacobian is, to
  // leading order, |h_i| * ||v||^2 / ||v|| = |h_i| * ||v|| for this
  // problem (F''  = 2 per coordinate, so the truncation term is exactly
  // h * v_i^2, the SAME algebraic fact test/num/newton_krylov_test.cpp's
  // own Mffd.ActionAgreesWithAnalyticJacobianAtTheWalkerPerniceStep
  // comment derives) -- so the two actions' own difference is bounded by
  // the SUM of the two truncation terms plus a round-off margin, never
  // tighter than |h_native - h_petsc| * max(||v||^2) when the two h
  // values genuinely differ, which the hard-coded derivative_gate.hpp
  // constant does not know about. This test derives its own bound from
  // the two MEASURED h values rather than reusing kMatrixFreeAgreementConstant
  // blind to them.
  const double max_v_norm = std::sqrt(2.0 * 3.0 * 3.0);  // the n_vectors=4 draws below are O(1)
  const double h_diff_term = std::fabs(native_h - petsc_h) * max_v_norm * max_v_norm;
  const double eps = std::numeric_limits<double>::epsilon();
  const double roundoff_margin =
      fuse::linalg::kMatrixFreeAgreementConstant * eps * norm_est.value() * max_v_norm;
  const double bound = h_diff_term + roundoff_margin;

  Result<MatrixFreeAgreementResult> agree =
      matrix_free_agreement(2, 2, petsc_apply, native_apply, /*n_vectors=*/4, norm_est.value());
  ASSERT_TRUE(agree.ok());
  std::fprintf(stderr,
              "[petsc.derivative_gate agreement] max_defect=%.17g derived_bound=%.17g "
              "builtin_bound=%.17g\n",
              agree.value().max_defect, bound, agree.value().bound);
  EXPECT_LE(agree.value().max_defect, bound)
      << "the PETSc and native MFFD actions disagree beyond the bound this test derives from "
         "their own measured h values -- a finding to report, not a tolerance to loosen";
}
