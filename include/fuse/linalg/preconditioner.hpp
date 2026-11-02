#pragma once

/// A preconditioner is the same shape as a `LinearOperator` (operator.hpp):
/// `apply(x, y)` computing `y <- M^-1 x`, an approximate solve rather than
/// an exact one. `pcg.hpp`'s and `gmres.hpp`'s solvers take the operator
/// and the preconditioner through two independent template parameters
/// constrained by two independent concepts, `LinearOperator` and
/// `Preconditioner`, so that any type satisfying the right shape is
/// accepted -- `IdentityPreconditioner` and `JacobiPreconditioner` below,
/// or a caller's own, with no base class and no virtual dispatch.
///
/// `JacobiPreconditioner` is built from `DistSparseMatrix::diagonal()`
/// (dist_sparse_matrix.hpp), which needs no communication (a participant's
/// own diagonal entries are, by that type's own column split, always
/// inside its own diagonal block) -- so building a Jacobi preconditioner
/// is itself a purely local operation, once, before any solve.

#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>

#include <fmt/format.h>

#include "fuse/base/error.hpp"
#include "fuse/base/result.hpp"
#include "fuse/linalg/comm.hpp"
#include "fuse/linalg/dist_sparse_matrix.hpp"
#include "fuse/linalg/dist_vector.hpp"

namespace fuse::linalg {

/// Any `T` for which `p.apply(x, y)` computes `y <- M^-1 x` -- structurally
/// identical to `LinearOperator` (operator.hpp), named separately because
/// the two mean different things to a reader even though nothing in the
/// language distinguishes them: `pcg`/`gmres` constrain their two template
/// parameters by the concept that matches each argument's ROLE, not by a
/// shape the compiler could not tell apart anyway. Carries the SAME
/// uniform-failure precondition `LinearOperator`'s own header comment
/// states (Review 2026-10-11 NL-6): a `p.apply` that can refuse on some
/// ranks of a call and succeed on others strands the solver's other
/// ranks inside a later collective. `IdentityPreconditioner` and
/// `JacobiPreconditioner` below never fail data-dependently, so this is
/// latent for every preconditioner this library builds itself.
template <class T>
concept Preconditioner = requires(const T& p, DistVector& x, DistVector& y) {
  { p.apply(x, y) } -> std::same_as<Result<void>>;
};

/// `M = I`: `apply` copies its input to its output and never fails. The
/// preconditioner a solver is handed when nothing better is known, and the
/// reference an actual preconditioner's effect (fewer iterations, not a
/// different converged answer) is measured against.
class IdentityPreconditioner {
 public:
  [[nodiscard]] Result<void> apply(DistVector& x, DistVector& y) const {
    y.copy_from(x);
    return Result<void>::success();
  }
};

/// `M = diag(A)`: `apply` scales each OWNED entry by the reciprocal of the
/// matching diagonal entry. Needs no ghost update and no reduction --
/// `DistSparseMatrix::diagonal()` already resolved every diagonal entry
/// locally, and scaling one owned entry by another is elementwise.
class JacobiPreconditioner {
 public:
  /// Builds from `a`'s diagonal. Refuses rather than dividing by zero:
  ///
  ///   * `E-LINALG-ZERO-DIAGONAL` -- some owned row's diagonal entry is
  ///     exactly zero, so no reciprocal exists for it. A structurally
  ///     absent diagonal entry (`DistSparseMatrix::diagonal()` reports
  ///     `0.0` for one, per that method's own documentation) is
  ///     indistinguishable here from an explicit zero and refused the same
  ///     way: either way `1/0` is not a number a solver can use.
  [[nodiscard]] static Result<JacobiPreconditioner> from_matrix(const DistSparseMatrix& a) {
    // The scratch DistVector below exists only to receive `diagonal()`'s
    // output; `diagonal()` issues no communication at all (the comment
    // above says why), so which exchanger/reducer it is built with is
    // immaterial -- the no-ghost, one-rank pair is the cheapest choice
    // that satisfies DistVector's constructor.
    DistVector d(a.local_rows(), make_serial_exchanger(), make_serial_reducer());
    a.diagonal(d);

    std::vector<double> inv_diag(a.local_rows());
    for (std::size_t i = 0; i < a.local_rows(); ++i) {
      const double diag_i = d.data()[i];
      if (diag_i == 0.0) {
        return Failure{"E-LINALG-ZERO-DIAGONAL",
                      fmt::format("JacobiPreconditioner::from_matrix: local row {} has a zero "
                                 "(or structurally absent) diagonal entry; no reciprocal exists",
                                 i)};
      }
      inv_diag[i] = 1.0 / diag_i;
    }
    return JacobiPreconditioner(std::move(inv_diag));
  }

  [[nodiscard]] Result<void> apply(DistVector& x, DistVector& y) const {
    FUSE_ASSERT(x.owned_size() == inv_diag_.size(),
               "JacobiPreconditioner::apply: x's owned size {} disagrees with the {} diagonal "
               "entries this preconditioner was built from",
               x.owned_size(), inv_diag_.size());
    FUSE_ASSERT(y.owned_size() == inv_diag_.size(),
               "JacobiPreconditioner::apply: y's owned size {} disagrees with the {} diagonal "
               "entries this preconditioner was built from",
               y.owned_size(), inv_diag_.size());
    for (std::size_t i = 0; i < inv_diag_.size(); ++i) {
      y.data()[i] = inv_diag_[i] * x.data()[i];
    }
    return Result<void>::success();
  }

 private:
  explicit JacobiPreconditioner(std::vector<double> inv_diag) : inv_diag_(std::move(inv_diag)) {}

  std::vector<double> inv_diag_;
};

/// Symmetric SOR (SSOR), relaxation `omega`, over THIS PARTICIPANT'S OWN
/// diagonal block alone (`DistSparseMatrix::local_diagonal_block()`) --
/// the off-process block is never read, which makes this BLOCK-JACOBI
/// ACROSS ranks (each rank preconditions only its own rows) with SSOR
/// INSIDE each block -- the standard parallel SSOR construction (no
/// cross-rank triangular solve is attempted; a true global SSOR would
/// need one, and is not what this class does). Symmetric (for a
/// symmetric `A`, the two triangular sweeps below compose to a symmetric
/// positive definite `M`, the standard SSOR result, e.g. Saad,
/// "Iterative Methods for Sparse Linear Systems", 2nd ed., Sec. 4.1) --
/// legal for `pcg` as well as `gmres`, unlike a plain (non-symmetric)
/// SOR sweep would be.
///
/// SPLITTING. `A = D + L + U` (`D` diagonal, `L`/`U` the strictly lower/
/// upper parts of the SAME local diagonal-block CSR). The preconditioner
/// matrix is `M = (D/omega + L) D^-1 (D/omega + U)`; `apply(x, y)` solves
/// `M y = x` by two triangular sweeps, never forming `M` itself:
///
///   (D/omega + L) z = x                      -- forward substitution,
///                                                rows 0 .. local_rows()-1
///   (D/omega + U) y = (D/omega) z             -- backward substitution,
///                                                rows local_rows()-1 .. 0
///
/// At `omega = 1` with `L = U = 0` (a diagonal `A`), `M = D`, so `apply`
/// reduces EXACTLY to `JacobiPreconditioner`'s own `y = D^-1 x` -- the
/// sanity check this class's own test fires.
class SsorPreconditioner {
 public:
  /// Builds from `a`'s own local diagonal block and `omega`. Refuses:
  ///
  ///   * `E-LINALG-SSOR-OMEGA-RANGE` -- `omega` is not in the OPEN
  ///     interval `(0, 2)`, the classical SOR convergence requirement
  ///     (Saad, Thm 4.5-ish; `omega <= 0` reverses the sweep's own sign,
  ///     `omega >= 2` over-relaxes past where `M` stays positive
  ///     definite for an SPD `A`).
  ///   * `E-LINALG-ZERO-DIAGONAL` -- the SAME refusal `JacobiPreconditioner`
  ///     makes (a zero or structurally absent diagonal entry has no
  ///     reciprocal), named identically since it is the identical problem.
  [[nodiscard]] static Result<SsorPreconditioner> from_matrix(const DistSparseMatrix& a, double omega = 1.0) {
    if (!(omega > 0.0 && omega < 2.0)) {
      return Failure{"E-LINALG-SSOR-OMEGA-RANGE",
                    fmt::format("SsorPreconditioner::from_matrix: omega {} is not in the open "
                               "interval (0, 2) SOR convergence requires",
                               omega)};
    }
    const auto block = a.local_diagonal_block();
    const std::size_t n = a.local_rows();
    std::vector<double> diag(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t p = block.row_ptr[i]; p < block.row_ptr[i + 1]; ++p) {
        if (block.col_idx[p] == i) diag[i] = block.val[p];
      }
      if (diag[i] == 0.0) {
        return Failure{"E-LINALG-ZERO-DIAGONAL",
                      fmt::format("SsorPreconditioner::from_matrix: local row {} has a zero (or "
                                 "structurally absent) diagonal entry; no reciprocal exists",
                                 i)};
      }
    }
    return SsorPreconditioner(std::vector<std::size_t>(block.row_ptr.begin(), block.row_ptr.end()),
                              std::vector<std::size_t>(block.col_idx.begin(), block.col_idx.end()),
                              std::vector<double>(block.val.begin(), block.val.end()), std::move(diag), omega);
  }

  [[nodiscard]] Result<void> apply(DistVector& x, DistVector& y) const {
    const std::size_t n = diag_.size();
    FUSE_ASSERT(x.owned_size() == n,
               "SsorPreconditioner::apply: x's owned size {} disagrees with the {} rows this "
               "preconditioner was built from",
               x.owned_size(), n);
    FUSE_ASSERT(y.owned_size() == n,
               "SsorPreconditioner::apply: y's owned size {} disagrees with the {} rows this "
               "preconditioner was built from",
               y.owned_size(), n);

    // Forward sweep: (D/omega + L) z = x, rows ascending (row i's own L
    // entries, column < i, are already-computed z values).
    std::vector<double> z(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      double rhs = x.data()[i];
      for (std::size_t p = row_ptr_[i]; p < row_ptr_[i + 1]; ++p) {
        if (col_idx_[p] < i) rhs -= val_[p] * z[col_idx_[p]];
      }
      z[i] = rhs * omega_ / diag_[i];
    }
    // Backward sweep: (D/omega + U) y = (D/omega) z, rows DESCENDING (row
    // i's own U entries, column > i, are already-computed y values).
    for (std::size_t ii = 0; ii < n; ++ii) {
      const std::size_t i = n - 1 - ii;
      double rhs = (diag_[i] / omega_) * z[i];
      for (std::size_t p = row_ptr_[i]; p < row_ptr_[i + 1]; ++p) {
        if (col_idx_[p] > i) rhs -= val_[p] * y.data()[col_idx_[p]];
      }
      y.data()[i] = rhs * omega_ / diag_[i];
    }
    return Result<void>::success();
  }

 private:
  SsorPreconditioner(std::vector<std::size_t> row_ptr, std::vector<std::size_t> col_idx, std::vector<double> val,
                     std::vector<double> diag, double omega)
      : row_ptr_(std::move(row_ptr)), col_idx_(std::move(col_idx)), val_(std::move(val)), diag_(std::move(diag)),
        omega_(omega) {}

  std::vector<std::size_t> row_ptr_;
  std::vector<std::size_t> col_idx_;
  std::vector<double> val_;
  std::vector<double> diag_;
  double omega_;
};

}  // namespace fuse::linalg
