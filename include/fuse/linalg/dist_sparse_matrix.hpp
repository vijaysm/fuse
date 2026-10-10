#pragma once

/// A sparse matrix split across participants by whole rows, each row held
/// in compressed (CSR) form split into two blocks: a DIAGONAL block (this
/// participant's own columns) and an OFF-PROCESS block (every other
/// column, which becomes one entry of the result vector's ghost region).
/// `apply()` is then exactly "update the input vector's ghosts, then two
/// local CSR products" -- no communication the matrix itself has to do,
/// because the vector's `GhostExchanger` already did it
/// (docs/dev/linalg-distributed.md has the full layout and the worked
/// example this comment only sketches).
///
/// Communication is injected (`fuse::linalg::Reducer`, carried for use by
/// operations built on top of this one, e.g. a matrix norm; nothing in
/// this file issues a new collective of its own -- see `from_triplets`'s
/// own comment for why), never included: this header depends on `base`
/// and `linalg::comm` alone, matching `linalg`'s position in
/// docs/dev/layering.md.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "fuse/base/result.hpp"
#include "fuse/linalg/comm.hpp"
#include "fuse/linalg/dist_vector.hpp"

namespace fuse::linalg {

/// A COO entry before canonicalisation, in GLOBAL row/column numbering.
/// `fuse::backend`'s own `Triplet` (src/backend/spacetime.cpp) is a
/// different, private type for a 32-bit, single-rank reconstruction
/// system; this one is `fuse::linalg`'s public, distributed-assembly
/// equivalent and the two are not interchangeable.
struct Triplet {
  std::int64_t row;
  std::int64_t col;
  double value;
};

/// How global rows (equivalently, since the operators this solves for are
/// square and partitioned identically on both sides, global columns) are
/// split across participants: `row_lo[r]` is the first row/column rank `r`
/// owns, ascending, one entry per participant. Rank `r`'s own row range is
/// `[row_lo[r], row_lo[r + 1])`, with the last participant's upper bound
/// being the matrix's `global_n` rather than a further `row_lo` entry.
/// Knowing every participant's `row_lo` up front is what lets
/// `from_triplets` resolve a column's owner with one `std::upper_bound`
/// and no communication, exactly as this type's task-level name promises.
struct RowOwnership {
  std::vector<std::int64_t> row_lo;  ///< size == number of participants, ascending
};

class DistSparseMatrix {
 public:
  /// Builds this participant's row block `[row_lo, row_hi)` of a
  /// `global_n x global_n` matrix from `t`, a COO list in global
  /// numbering. Every triplet must name a row in `[row_lo, row_hi)` --
  /// `t` is this participant's OWN contribution, not the whole matrix --
  /// or the triplet is refused (see below); duplicate `(row, col)` pairs
  /// are summed, matching the usual COO-to-CSR convention. The column
  /// split (diagonal vs. off-process) and the off-process block's
  /// `GhostPattern` are DERIVED from `t` and `owners`, never supplied by
  /// the caller.
  ///
  /// `reducer` is stored for an operation built on this one to use (a
  /// global matrix norm, say) -- NOT spent here. `from_triplets` issues no
  /// collective of its own: every check below is answerable from this
  /// participant's own `t` and `owners`, which the standing rule that a
  /// collective must never be skipped on a locally computed condition
  /// would otherwise put at risk if a new collective were added and then
  /// short-circuited by one rank's local refusal. Avoided by not adding
  /// one at all, not by being careful with one that exists.
  ///
  /// Errors, never throws:
  ///   * `E-LINALG-FOREIGN-ROW` -- a triplet names a row outside
  ///     `[row_lo, row_hi)`; this participant's own `t` must be exactly
  ///     its own rows, nothing a peer's row could ever satisfy by being
  ///     retried elsewhere.
  ///   * `E-LINALG-COL-RANGE` -- a triplet names a column outside
  ///     `[0, global_n)`, which no `RowOwnership` can resolve an owner for.
  [[nodiscard]] static Result<DistSparseMatrix> from_triplets(
      std::int64_t row_lo, std::int64_t row_hi, std::int64_t global_n,
      const std::vector<Triplet>& t, std::shared_ptr<const Reducer> reducer,
      const RowOwnership& owners);

  /// What a `DistVector` passed to `apply()` as `x` must carry as its
  /// ghost region: exactly the off-process columns `from_triplets` found,
  /// in the order `apply()`'s off-diagonal product reads them.
  [[nodiscard]] const GhostPattern& ghost_pattern() const noexcept { return ghost_pattern_; }

  /// `y <- A x`. Calls `x.update_ghosts()` first (COLLECTIVE under MPI --
  /// see that method's own documentation), then the diagonal and
  /// off-diagonal CSR products, which need no further communication.
  /// `x` and `y` must both carry `local_rows()` owned entries, and `x`
  /// must carry this matrix's own `ghost_pattern()` (a shape mismatch
  /// there is a caller mistake and throws, the same split
  /// `DistVector::check_same_distribution` draws elsewhere in this
  /// library); a communication failure inside `update_ghosts()` is the
  /// one outcome this returns a `Failure` for.
  [[nodiscard]] Result<void> apply(DistVector& x, DistVector& y) const;

  /// `d.data()[i] <- A(row_lo + i, row_lo + i)` (0.0 where the diagonal
  /// entry is structurally absent), for `i` in `[0, local_rows())`. Needs
  /// no communication: a participant's own diagonal entries are always
  /// inside its own diagonal block, by the column split's own definition.
  void diagonal(DistVector& d) const;

  [[nodiscard]] std::size_t local_rows() const noexcept { return local_rows_; }
  [[nodiscard]] std::int64_t global_rows() const noexcept { return global_n_; }

 private:
  DistSparseMatrix() = default;

  std::int64_t row_lo_ = 0;
  std::int64_t global_n_ = 0;
  std::size_t local_rows_ = 0;
  std::shared_ptr<const Reducer> reducer_;

  // Diagonal block: CSR over LOCAL row/column indices, both in [0, local_rows_).
  std::vector<std::size_t> diag_row_ptr_;
  std::vector<std::size_t> diag_col_idx_;
  std::vector<double> diag_val_;

  // Off-process block: CSR over local rows, columns indexing the GHOST
  // region (0, ghost_pattern_.owner_rank.size()), not the owned block.
  std::vector<std::size_t> off_row_ptr_;
  std::vector<std::size_t> off_col_idx_;
  std::vector<double> off_val_;

  GhostPattern ghost_pattern_;
};

}  // namespace fuse::linalg
