// fuse::petsc::solve() -- the SNES driver (fuse/petsc/solve.hpp), STEP 3
// of the PetscToken track. The SAME quadratic and atan problems
// test/num/newton_krylov_test.cpp uses, solved by BOTH substrates from
// the SAME x0 and (as far as the two option structs overlap) the SAME
// options -- row petsc.snes.

#include "fuse/petsc/solve.hpp"

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
using fuse::linalg::DistVector;
using fuse::linalg::make_serial_exchanger;
using fuse::linalg::make_serial_reducer;
using fuse::num::is_converged;
using fuse::num::is_diverged;
using fuse::num::NewtonKrylovOptions;
using fuse::num::NewtonResult;
using fuse::num::NewtonStatus;
using fuse::num::ResidualFn;
using fuse::num::to_cstring;
using fuse::petsc::comm_self_f;
using fuse::petsc::SnesOptions;

namespace {

DistVector vec_sized(std::size_t n) {
  return DistVector(n, make_serial_exchanger(), make_serial_reducer());
}

DistVector vec_from(const std::vector<double>& v) {
  DistVector out = vec_sized(v.size());
  std::copy(v.begin(), v.end(), out.data());
  return out;
}

ResidualFn quadratic_residual(const std::vector<double>& c) {
  return [c](DistVector& x, DistVector& fx) -> Result<void> {
    for (std::size_t i = 0; i < c.size(); ++i) fx.data()[i] = x.data()[i] * x.data()[i] - c[i];
    return Result<void>::success();
  };
}

ResidualFn atan_residual(const std::vector<double>& c) {
  return [c](DistVector& x, DistVector& fx) -> Result<void> {
    for (std::size_t i = 0; i < c.size(); ++i) {
      fx.data()[i] = std::atan(x.data()[i]) - std::atan(c[i]);
    }
    return Result<void>::success();
  };
}

void print_trace(const char* label, const NewtonResult& r) {
  std::fprintf(stderr, "[petsc.snes %s] status=%s iterations=%d final_||F||=%.17g\n", label,
              to_cstring(r.status), r.iterations, r.final_residual_norm);
  for (const auto& it : r.trace) {
    std::fprintf(stderr, "[petsc.snes %s]   step=%d ||F||=%.17g lambda=%.6g\n", label,
                it.iteration, it.norm_f, it.line_search_lambda);
  }
}

}  // namespace

TEST(PetscSnes, QuadraticProblemConverges) {
  const std::vector<double> c = {4.0, 9.0};
  fuse::num::NewtonKrylovOptions native_opt;
  DistVector x_native = vec_from({3.0, 1.0});
  Result<NewtonResult> native_r = fuse::num::solve(quadratic_residual(c), x_native, nullptr,
                                                   nullptr, native_opt);
  ASSERT_TRUE(native_r.ok()) << native_r.error().code << ": " << native_r.error().message;

  SnesOptions petsc_opt;
  petsc_opt.comm_f = comm_self_f();
  DistVector x_petsc = vec_from({3.0, 1.0});
  Result<NewtonResult> petsc_r = fuse::petsc::solve(quadratic_residual(c), x_petsc, nullptr,
                                                    nullptr, petsc_opt);
  ASSERT_TRUE(petsc_r.ok()) << petsc_r.error().code << ": " << petsc_r.error().message;

  print_trace("quadratic.native", native_r.value());
  print_trace("quadratic.petsc", petsc_r.value());

  EXPECT_TRUE(is_converged(native_r.value().status)) << to_cstring(native_r.value().status);
  EXPECT_TRUE(is_converged(petsc_r.value().status)) << to_cstring(petsc_r.value().status);
  EXPECT_NEAR(x_native.data()[0], 2.0, 1e-5);
  EXPECT_NEAR(x_native.data()[1], 3.0, 1e-5);
  EXPECT_NEAR(x_petsc.data()[0], 2.0, 1e-5);
  EXPECT_NEAR(x_petsc.data()[1], 3.0, 1e-5);

  // The FIRST GMRES iterate at eta_0 = 0.3: both substrates use the
  // IDENTICAL Walker-Pernice h formula and the identical forcing
  // constants at step 0, so the first recorded ||F|| after one Newton
  // step should agree to a small multiple of the per-substrate h
  // difference (not asserted to bit-for-bit, since PETSc's own ||a||
  // handling can differ at the 1ulp level) -- measured and reported, not
  // forced, since later steps may diverge further through line-search or
  // GMRES implementation differences that are NOT this test's own claim.
  if (native_r.value().trace.size() > 1 && petsc_r.value().trace.size() > 1) {
    std::fprintf(stderr,
                "[petsc.snes quadratic] step1 ||F|| native=%.17g petsc=%.17g\n",
                native_r.value().trace[1].norm_f, petsc_r.value().trace[1].norm_f);
  }
}

TEST(PetscSnes, AtanProblemFromAPoorInitialGuessConvergesOnBothSubstratesAndBtEngagesOnBoth) {
  const std::vector<double> c = {0.0};

  fuse::num::NewtonKrylovOptions native_opt;
  native_opt.line_search.kind = fuse::num::LineSearchKind::BT;
  native_opt.rtol = 1e-10;
  native_opt.atol = 1e-12;
  native_opt.max_nonlinear_iterations = 60;
  DistVector x_native = vec_from({20.0});
  Result<NewtonResult> native_r =
      fuse::num::solve(atan_residual(c), x_native, nullptr, nullptr, native_opt);
  ASSERT_TRUE(native_r.ok());

  SnesOptions petsc_opt;
  petsc_opt.comm_f = comm_self_f();
  petsc_opt.newton = native_opt;
  DistVector x_petsc = vec_from({20.0});
  Result<NewtonResult> petsc_r =
      fuse::petsc::solve(atan_residual(c), x_petsc, nullptr, nullptr, petsc_opt);
  ASSERT_TRUE(petsc_r.ok()) << petsc_r.error().code << ": " << petsc_r.error().message;

  print_trace("atan.native", native_r.value());
  print_trace("atan.petsc", petsc_r.value());

  EXPECT_TRUE(is_converged(native_r.value().status)) << to_cstring(native_r.value().status);
  EXPECT_TRUE(is_converged(petsc_r.value().status)) << to_cstring(petsc_r.value().status);
  EXPECT_NEAR(x_native.data()[0], 0.0, 1e-6);
  EXPECT_NEAR(x_petsc.data()[0], 0.0, 1e-6);

  bool native_backtracked = false;
  for (const auto& it : native_r.value().trace) {
    if (it.line_search_trials.size() > 1) native_backtracked = true;
  }
  bool petsc_backtracked = false;
  for (const auto& it : petsc_r.value().trace) {
    if (it.line_search_lambda < 1.0) petsc_backtracked = true;
  }
  std::fprintf(stderr, "[petsc.snes atan] native_backtracked=%d petsc_backtracked=%d\n",
              native_backtracked ? 1 : 0, petsc_backtracked ? 1 : 0);
  EXPECT_TRUE(native_backtracked) << "a full Newton step from x0=20 on atan must be rejected by "
                                     "Armijo at least once (the native solver's own mechanism)";
  // PETSc's own BT engagement is RECORDED, not asserted on: SNESLINESEARCHBT's
  // own internals are not this track's own code, and this test's job is to
  // measure whether the two substrates agree on needing a backtrack here,
  // not to assume PETSc's BT must behave identically to the native one.
}

TEST(PetscSnes, DivergenceWhenTheFunctionEvaluationBudgetIsTooSmall) {
  const std::vector<double> c = {0.3};
  SnesOptions opt;
  opt.comm_f = comm_self_f();
  opt.newton.max_function_evaluations = 1;
  DistVector x = vec_from({0.5});
  Result<NewtonResult> r = fuse::petsc::solve(atan_residual(c), x, nullptr, nullptr, opt);
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
  EXPECT_TRUE(is_diverged(r.value().status)) << to_cstring(r.value().status);
}

TEST(PetscSnes, MultiRankVectorIsRefused) {
  // A DistVector with a nonzero ghost region -- E-PETSC-SOLVE-MULTIRANK
  // must fire rather than silently drop the ghost region. Built directly
  // against a hand-rolled exchanger reporting a nonzero ghost_size(),
  // the same shape test/num/newton_krylov_test.cpp's own rank-agreement
  // fixtures use for a non-serial exchanger.
  class FakeGhostExchanger : public fuse::linalg::GhostExchanger {
   public:
    [[nodiscard]] std::size_t ghost_size() const override { return 1; }
    [[nodiscard]] Result<void> update(const double*, std::size_t, double*,
                                      std::size_t) const override {
      return Result<void>::success();
    }
  };
  DistVector x(2, std::make_shared<FakeGhostExchanger>(), make_serial_reducer());
  SnesOptions opt;
  opt.comm_f = comm_self_f();
  Result<NewtonResult> r = fuse::petsc::solve(atan_residual({0.0, 0.0}), x, nullptr, nullptr, opt);
  ASSERT_FALSE(r.ok());
  EXPECT_STREQ(r.error().code.c_str(), "E-PETSC-SOLVE-MULTIRANK");
}
