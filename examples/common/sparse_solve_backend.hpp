#pragma once
/// \file sparse_solve_backend.hpp
/// One sparse direct LU solver for the examples, with the same five calls
/// `Eigen::SparseLU` exposes -- `analyzePattern`, `factorize`, `compute`,
/// `solve`, `info` -- so `StokesSolver`'s `CnStepLuCache::lu` and
/// `MonolithicGrayScott`'s `jac_solver_` can hold one with no other call-site
/// change.
///
/// Two builds of the same class:
///
/// * `FUSE_HAVE_CUSPARSE == 0` (the default, `cmake -DFUSE_ENABLE_CUSPARSE=OFF`):
///   a one-member wrapper around `Eigen::SparseLU<Eigen::SparseMatrix<double>>`.
///   Every call forwards to the same Eigen method the call sites used to call
///   directly, on the same object type, so results are bit for bit what they
///   were before the swap (`test/verification/sparse_solve_backend_test.cpp`
///   checks exactly that against a bare `Eigen::SparseLU`).
///
/// * `FUSE_HAVE_CUSPARSE == 1` (`-DFUSE_ENABLE_CUSPARSE=ON`, needs a CUDA
///   Kokkos): the numeric factorisation and the solves run in cusolverRf on
///   the device. cusolverRf is NVIDIA's *re*-factorisation API -- it takes a
///   first LU (with its pivoting permutations) computed elsewhere and then,
///   for any later matrix with the same pattern, redoes only the numeric
///   factorisation and the triangular solves on the GPU, reusing the pivot
///   sequence. That is precisely the shape of both call sites after Phase 1
///   (`39ad7a7`, `2e9db1f`): one pattern, repeated `factorize()` with new
///   values. The seed LU comes from cusolverSp's host LU
///   (`cusolverSp_LOWLEVEL_PREVIEW.h`), run once per pattern, after an AMD
///   fill-reducing symmetric permutation computed with Eigen's own
///   `AMDOrdering` (cusolverSp's host LU does no reordering of its own; on a
///   2D grid Jacobian that would mean banded fill). Results are NOT bitwise
///   Eigen's -- different pivot sequence, different ordering, FMA on the
///   device -- the example tests' own tolerances are the acceptance bound.
///
/// Honest expectation for the matrices the examples actually build (a
/// 128-unknown Gray-Scott Jacobian at `kN = 8`; a few-thousand-unknown Stokes
/// saddle point): each cusolverRf call is a handful of kernel launches plus
/// two host<->device copies, which costs more than Eigen's whole host
/// factorisation at these sizes. The device path is here to be measured, not
/// assumed faster; `docs/dev/runner-device-dispatch.md` carries the numbers.

#include <Eigen/OrderingMethods>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

#include <string_view>

#include "fuse/base/config.hpp"

#if FUSE_HAVE_CUSPARSE
#include <cuda_runtime.h>
#include <cusolverRf.h>
#include <cusolverSp.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>
#include <cusparse.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>
#endif

namespace fuse_examples {

#if !FUSE_HAVE_CUSPARSE

class SparseLuBackend {
 public:
  using Matrix = Eigen::SparseMatrix<double>;

  [[nodiscard]] static constexpr std::string_view name() noexcept { return "Eigen::SparseLU"; }

  void analyzePattern(const Matrix& A) { lu_.analyzePattern(A); }
  void factorize(const Matrix& A) { lu_.factorize(A); }
  void compute(const Matrix& A) { lu_.compute(A); }
  [[nodiscard]] Eigen::VectorXd solve(const Eigen::VectorXd& b) const { return lu_.solve(b); }
  [[nodiscard]] Eigen::ComputationInfo info() const { return lu_.info(); }

 private:
  Eigen::SparseLU<Matrix> lu_;
};

#else  // FUSE_HAVE_CUSPARSE

class SparseLuBackend {
 public:
  using Matrix = Eigen::SparseMatrix<double>;

  [[nodiscard]] static constexpr std::string_view name() noexcept { return "cusolverRf"; }

  SparseLuBackend() = default;
  SparseLuBackend(const SparseLuBackend&) = delete;
  SparseLuBackend& operator=(const SparseLuBackend&) = delete;
  ~SparseLuBackend() { release_all(); }

  /// Fixes the pattern: the AMD permutation and the permuted CSR structure.
  /// Any cusolverRf state from a previous pattern is discarded.
  void analyzePattern(const Matrix& A) {
    release_rf();
    n_ = static_cast<int>(A.rows());
    if (A.rows() != A.cols() || n_ == 0) {
      pattern_ok_ = false;
      info_ = Eigen::InvalidInput;
      return;
    }
    Eigen::AMDOrdering<int> ordering;
    ordering(A, perm_);  // symmetrises the pattern itself (A + A^T)
    permute_to_csr(A);
    row_.assign(csr_.outerIndexPtr(), csr_.outerIndexPtr() + n_ + 1);
    col_.assign(csr_.innerIndexPtr(), csr_.innerIndexPtr() + csr_.nonZeros());
    pattern_ok_ = true;
    info_ = Eigen::InvalidInput;  // no numeric factors yet
  }

  /// Numeric factorisation of A on the pattern fixed by analyzePattern().
  /// First call per pattern: cusolverSp host LU -> cusolverRf setup; later
  /// calls: cusolverRfResetValues + cusolverRfRefactor on the device only.
  void factorize(const Matrix& A) {
    if (!pattern_ok_ || A.rows() != n_ || A.cols() != n_) {
      info_ = Eigen::InvalidInput;
      return;
    }
    permute_to_csr(A);
    if (!same_pattern()) {
      // The caller changed the structure under us (explicit zeros pruned
      // differently, say): the only safe thing is a fresh seed.
      release_rf();
      row_.assign(csr_.outerIndexPtr(), csr_.outerIndexPtr() + n_ + 1);
      col_.assign(csr_.innerIndexPtr(), csr_.innerIndexPtr() + csr_.nonZeros());
    }
    if (!rf_ok_ && !seed_from_host_lu()) return;  // info_ already set
    const int nnz = static_cast<int>(csr_.nonZeros());
    cuda_check(cudaMemcpy(d_val_, csr_.valuePtr(), sizeof(double) * nnz, cudaMemcpyHostToDevice),
               "cudaMemcpy(values)");
    cusolverStatus_t st = cusolverRfResetValues(n_, nnz, d_row_, d_col_, d_val_, d_P_, d_Q_, rf_);
    if (st != CUSOLVER_STATUS_SUCCESS) { numeric_failure(st, "cusolverRfResetValues"); return; }
    st = cusolverRfRefactor(rf_);
    if (st != CUSOLVER_STATUS_SUCCESS) { numeric_failure(st, "cusolverRfRefactor"); return; }
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(refactor)");
    info_ = Eigen::Success;
  }

  void compute(const Matrix& A) {
    analyzePattern(A);
    if (pattern_ok_) factorize(A);
  }

  /// x = A^{-1} b. Requires a successful factorize(); otherwise returns a
  /// zero vector and leaves info() reporting the earlier failure.
  [[nodiscard]] Eigen::VectorXd solve(const Eigen::VectorXd& b) const {
    if (info_ != Eigen::Success || !rf_ok_ || b.size() != n_) {
      if (info_ == Eigen::Success) info_ = Eigen::InvalidInput;
      return Eigen::VectorXd::Zero(n_);
    }
    // B = P A P^T with B y = P b, x = P^T y.
    const Eigen::VectorXd pb = perm_ * b;
    Eigen::VectorXd y(n_);
    cuda_check(cudaMemcpy(d_x_, pb.data(), sizeof(double) * n_, cudaMemcpyHostToDevice),
               "cudaMemcpy(rhs)");
    const cusolverStatus_t st = cusolverRfSolve(rf_, d_P_, d_Q_, 1, d_tmp_, n_, d_x_, n_);
    if (st != CUSOLVER_STATUS_SUCCESS) {
      info_ = Eigen::NumericalIssue;
      return Eigen::VectorXd::Zero(n_);
    }
    cuda_check(cudaMemcpy(y.data(), d_x_, sizeof(double) * n_, cudaMemcpyDeviceToHost),
               "cudaMemcpy(solution)");
    // One step of iterative refinement: r = Pb - B y on the host in extended
    // precision, dy = B^{-1} r on the device, y += dy. One host SpMV and one
    // more device solve. Why: the raw cusolverRf solve is backward-stable to
    // ~5e-16 on the Stokes saddle points, same as Eigen, but its FORWARD
    // error is a different kappa*eps-sized number than Eigen's (the two
    // solutions differed by up to 1e-8 relative over 24,519 real
    // fsi_transient solves, and an in-situ check saw ~4% of same-rhs repeats
    // differ when other solves ran in between -- not reproduced in isolation,
    // where back-to-back repeats were bitwise equal). A fixed-point coupling
    // iteration converges through a deterministic rounding floor and stalls
    // on one that moves (fsi_transient's fluid_force criterion stuck at
    // 2.3e-9 vs 1e-9). Refinement with an extended-precision residual pins
    // the answer to the exact solution rather than to either solver's
    // rounding: on the real 196- and 900-unknown systems the result is within
    // 3.7e-17 / 1.4e-16 of a long-double reference (Eigen double: 2.7e-14 /
    // 3.4e-14), repeats agree to working precision with or without other
    // solves interleaved, and fsi_transient passes with the flag on.
    {
      // The residual in extended precision (x86 long double: 64-bit mantissa).
      // Computed in double it would itself carry rounding of order eps*|B||y|
      // that depends on y's jittering low bits, and kappa amplifies that back
      // to ~1e-8 -- fsi_transient passed with a double residual, the energy
      // budget and example.fsi_channel did not. Mixed-precision refinement is
      // the textbook remedy: the limit is kappa*eps_ld, four orders lower.
      Eigen::VectorXd r(n_);
      const int*    rp = csr_.outerIndexPtr();
      const int*    ci = csr_.innerIndexPtr();
      const double* va = csr_.valuePtr();
      for (int i = 0; i < n_; ++i) {
        long double acc = static_cast<long double>(pb[i]);
        for (int k = rp[i]; k < rp[i + 1]; ++k) {
          acc -= static_cast<long double>(va[k]) * static_cast<long double>(y[ci[k]]);
        }
        r[i] = static_cast<double>(acc);
      }
      cuda_check(cudaMemcpy(d_x_, r.data(), sizeof(double) * n_, cudaMemcpyHostToDevice),
                 "cudaMemcpy(refine rhs)");
      const cusolverStatus_t st2 = cusolverRfSolve(rf_, d_P_, d_Q_, 1, d_tmp_, n_, d_x_, n_);
      if (st2 != CUSOLVER_STATUS_SUCCESS) {
        info_ = Eigen::NumericalIssue;
        return Eigen::VectorXd::Zero(n_);
      }
      cuda_check(cudaMemcpy(r.data(), d_x_, sizeof(double) * n_, cudaMemcpyDeviceToHost),
                 "cudaMemcpy(refine correction)");
      y += r;
    }
    return perm_.inverse() * y;
  }

  [[nodiscard]] Eigen::ComputationInfo info() const { return info_; }

 private:
  using Perm = Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int>;
  using Csr = Eigen::SparseMatrix<double, Eigen::RowMajor, int>;

  void permute_to_csr(const Matrix& A) {
    const Matrix PA = perm_ * A;
    const Matrix B = PA * perm_.inverse();
    csr_ = B;  // ColMajor -> RowMajor: cusolverRf wants CSR
    csr_.makeCompressed();
  }

  [[nodiscard]] bool same_pattern() const {
    const int nnz = static_cast<int>(csr_.nonZeros());
    return static_cast<int>(col_.size()) == nnz &&
           std::equal(row_.begin(), row_.end(), csr_.outerIndexPtr()) &&
           std::equal(col_.begin(), col_.end(), csr_.innerIndexPtr());
  }

  static void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
      throw std::runtime_error(std::string("SparseLuBackend: ") + what + ": " +
                               cudaGetErrorString(e));
    }
  }
  static void solver_check(cusolverStatus_t s, const char* what) {
    if (s != CUSOLVER_STATUS_SUCCESS) {
      throw std::runtime_error(std::string("SparseLuBackend: ") + what +
                               " failed with cusolverStatus_t " + std::to_string(static_cast<int>(s)));
    }
  }
  static void sparse_check(cusparseStatus_t s, const char* what) {
    if (s != CUSPARSE_STATUS_SUCCESS) {
      throw std::runtime_error(std::string("SparseLuBackend: ") + what +
                               " failed with cusparseStatus_t " + std::to_string(static_cast<int>(s)));
    }
  }
  void numeric_failure(cusolverStatus_t s, const char* what) {
    // A zero pivot is a property of the matrix, reported through info() like
    // Eigen does; anything else is an infrastructure failure and throws.
    if (s == CUSOLVER_STATUS_ZERO_PIVOT) {
      info_ = Eigen::NumericalIssue;
      return;
    }
    solver_check(s, what);
  }

  /// One host LU of the current csr_ with cusolverSp (partial pivoting,
  /// threshold 1.0), then hand L, U, P, Q to a fresh cusolverRf handle and
  /// upload the pattern. Returns false (with info_ set) on a zero pivot.
  bool seed_from_host_lu() {
    const int n = n_;
    const int nnz = static_cast<int>(csr_.nonZeros());
    if (sp_ == nullptr) solver_check(cusolverSpCreate(&sp_), "cusolverSpCreate");
    if (descr_ == nullptr) {
      sparse_check(cusparseCreateMatDescr(&descr_), "cusparseCreateMatDescr");
      cusparseSetMatType(descr_, CUSPARSE_MATRIX_TYPE_GENERAL);
      cusparseSetMatIndexBase(descr_, CUSPARSE_INDEX_BASE_ZERO);
    }

    csrluInfoHost_t lu_info = nullptr;
    solver_check(cusolverSpCreateCsrluInfoHost(&lu_info), "cusolverSpCreateCsrluInfoHost");
    struct InfoGuard {
      csrluInfoHost_t h;
      ~InfoGuard() { if (h) cusolverSpDestroyCsrluInfoHost(h); }
    } info_guard{lu_info};

    solver_check(cusolverSpXcsrluAnalysisHost(sp_, n, nnz, descr_, row_.data(), col_.data(), lu_info),
                 "cusolverSpXcsrluAnalysisHost");
    std::size_t internal_bytes = 0, workspace_bytes = 0;
    solver_check(cusolverSpDcsrluBufferInfoHost(sp_, n, nnz, descr_, csr_.valuePtr(), row_.data(),
                                                col_.data(), lu_info, &internal_bytes, &workspace_bytes),
                 "cusolverSpDcsrluBufferInfoHost");
    std::vector<char> workspace(std::max<std::size_t>(workspace_bytes, 1));
    solver_check(cusolverSpDcsrluFactorHost(sp_, n, nnz, descr_, csr_.valuePtr(), row_.data(),
                                            col_.data(), lu_info, /*pivot_threshold=*/1.0,
                                            workspace.data()),
                 "cusolverSpDcsrluFactorHost");
    int singular_at = -1;
    const cusolverStatus_t zp = cusolverSpDcsrluZeroPivotHost(sp_, lu_info, /*tol=*/0.0, &singular_at);
    if (zp == CUSOLVER_STATUS_ZERO_PIVOT || singular_at >= 0) {
      info_ = Eigen::NumericalIssue;
      return false;
    }
    solver_check(zp, "cusolverSpDcsrluZeroPivotHost");

    int nnzL = 0, nnzU = 0;
    solver_check(cusolverSpXcsrluNnzHost(sp_, &nnzL, &nnzU, lu_info), "cusolverSpXcsrluNnzHost");
    std::vector<int> P(n), Q(n), rowL(n + 1), colL(nnzL), rowU(n + 1), colU(nnzU);
    std::vector<double> valL(nnzL), valU(nnzU);
    solver_check(cusolverSpDcsrluExtractHost(sp_, P.data(), Q.data(),
                                             descr_, valL.data(), rowL.data(), colL.data(),
                                             descr_, valU.data(), rowU.data(), colU.data(),
                                             lu_info, workspace.data()),
                 "cusolverSpDcsrluExtractHost");

    // Does the extracted L carry its unit diagonal explicitly? Decide from the
    // data rather than from documentation: look for (0,0) in row 0.
    bool l_diag_stored = false;
    for (int k = rowL[0]; k < rowL[1]; ++k) {
      if (colL[k] == 0) { l_diag_stored = true; break; }
    }

    solver_check(cusolverRfCreate(&rf_), "cusolverRfCreate");
    solver_check(cusolverRfSetMatrixFormat(rf_, CUSOLVERRF_MATRIX_FORMAT_CSR,
                                           l_diag_stored ? CUSOLVERRF_UNIT_DIAGONAL_STORED_L
                                                         : CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_L),
                 "cusolverRfSetMatrixFormat");
    // Values are not const in cusolverRf's signature; it reads them only.
    std::vector<double> valA(csr_.valuePtr(), csr_.valuePtr() + nnz);
    solver_check(cusolverRfSetupHost(n, nnz, row_.data(), col_.data(), valA.data(),
                                     nnzL, rowL.data(), colL.data(), valL.data(),
                                     nnzU, rowU.data(), colU.data(), valU.data(),
                                     P.data(), Q.data(), rf_),
                 "cusolverRfSetupHost");
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(setup)");
    solver_check(cusolverRfAnalyze(rf_), "cusolverRfAnalyze");

    cuda_check(cudaMalloc(&d_row_, sizeof(int) * (n + 1)), "cudaMalloc(row)");
    cuda_check(cudaMalloc(&d_col_, sizeof(int) * std::max(nnz, 1)), "cudaMalloc(col)");
    cuda_check(cudaMalloc(&d_val_, sizeof(double) * std::max(nnz, 1)), "cudaMalloc(val)");
    cuda_check(cudaMalloc(&d_P_, sizeof(int) * n), "cudaMalloc(P)");
    cuda_check(cudaMalloc(&d_Q_, sizeof(int) * n), "cudaMalloc(Q)");
    cuda_check(cudaMalloc(&d_x_, sizeof(double) * n), "cudaMalloc(x)");
    cuda_check(cudaMalloc(&d_tmp_, sizeof(double) * n), "cudaMalloc(tmp)");
    cuda_check(cudaMemcpy(d_row_, row_.data(), sizeof(int) * (n + 1), cudaMemcpyHostToDevice), "cudaMemcpy(row)");
    cuda_check(cudaMemcpy(d_col_, col_.data(), sizeof(int) * nnz, cudaMemcpyHostToDevice), "cudaMemcpy(col)");
    cuda_check(cudaMemcpy(d_P_, P.data(), sizeof(int) * n, cudaMemcpyHostToDevice), "cudaMemcpy(P)");
    cuda_check(cudaMemcpy(d_Q_, Q.data(), sizeof(int) * n, cudaMemcpyHostToDevice), "cudaMemcpy(Q)");
    rf_ok_ = true;
    return true;
  }

  void release_rf() {
    if (rf_) { cusolverRfDestroy(rf_); rf_ = nullptr; }
    for (void** p : {reinterpret_cast<void**>(&d_row_), reinterpret_cast<void**>(&d_col_),
                     reinterpret_cast<void**>(&d_val_), reinterpret_cast<void**>(&d_P_),
                     reinterpret_cast<void**>(&d_Q_), reinterpret_cast<void**>(&d_x_),
                     reinterpret_cast<void**>(&d_tmp_)}) {
      if (*p) { cudaFree(*p); *p = nullptr; }
    }
    rf_ok_ = false;
  }
  void release_all() {
    release_rf();
    if (descr_) { cusparseDestroyMatDescr(descr_); descr_ = nullptr; }
    if (sp_) { cusolverSpDestroy(sp_); sp_ = nullptr; }
  }

  int n_ = 0;
  bool pattern_ok_ = false;
  bool rf_ok_ = false;
  mutable Eigen::ComputationInfo info_ = Eigen::InvalidInput;

  Perm perm_;                 // B = perm_ * A * perm_^T
  Csr csr_;                   // B in CSR, values refreshed per factorize()
  std::vector<int> row_, col_;  // B's pattern as seeded into cusolverRf

  cusolverSpHandle_t sp_ = nullptr;
  cusparseMatDescr_t descr_ = nullptr;
  cusolverRfHandle_t rf_ = nullptr;
  int* d_row_ = nullptr;
  int* d_col_ = nullptr;
  double* d_val_ = nullptr;
  int* d_P_ = nullptr;
  int* d_Q_ = nullptr;
  double* d_x_ = nullptr;
  double* d_tmp_ = nullptr;
};

#endif  // FUSE_HAVE_CUSPARSE

}  // namespace fuse_examples
