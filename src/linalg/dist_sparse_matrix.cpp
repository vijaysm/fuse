#include "fuse/linalg/dist_sparse_matrix.hpp"

#include <algorithm>
#include <functional>
#include <map>
#include <tuple>

#include <fmt/format.h>

#include "fuse/base/error.hpp"

namespace fuse::linalg {

namespace {

/// The owning participant of global column/row `idx`, found with one
/// `std::upper_bound` over `owners.row_lo` -- no communication, exactly
/// the property `RowOwnership`'s own header comment names.
int owner_of(std::int64_t idx, const RowOwnership& owners) {
  const auto it = std::upper_bound(owners.row_lo.begin(), owners.row_lo.end(), idx);
  return static_cast<int>(std::distance(owners.row_lo.begin(), it) - 1);
}

/// One (local row, local column, value) entry before CSR assembly, local
/// column meaning either a diagonal-block owned column or an off-process
/// block's ghost index, depending on which vector this lands in.
struct LocalEntry {
  std::size_t row;
  std::size_t col;
  double value;
};

/// Sorts `entries` by (row, col), sums duplicates (I5: the usual
/// COO-to-CSR convention), and writes the three CSR arrays. `entries` is
/// consumed (sorted in place) since nothing else needs the COO order
/// afterwards.
void assemble_csr(std::vector<LocalEntry>& entries, std::size_t n_rows,
                  std::vector<std::size_t>& row_ptr, std::vector<std::size_t>& col_idx,
                  std::vector<double>& val) {
  std::sort(entries.begin(), entries.end(), [](const LocalEntry& a, const LocalEntry& b) {
    return std::tie(a.row, a.col) < std::tie(b.row, b.col);
  });
  row_ptr.assign(n_rows + 1, 0);
  col_idx.clear();
  val.clear();
  col_idx.reserve(entries.size());
  val.reserve(entries.size());
  std::size_t i = 0;
  while (i < entries.size()) {
    std::size_t j = i + 1;
    double sum = entries[i].value;
    while (j < entries.size() && entries[j].row == entries[i].row &&
          entries[j].col == entries[i].col) {
      sum += entries[j].value;
      ++j;
    }
    col_idx.push_back(entries[i].col);
    val.push_back(sum);
    ++row_ptr[entries[i].row + 1];
    i = j;
  }
  for (std::size_t r = 0; r < n_rows; ++r) row_ptr[r + 1] += row_ptr[r];
}

}  // namespace

Result<DistSparseMatrix> DistSparseMatrix::from_triplets(std::int64_t row_lo,
                                                          std::int64_t row_hi,
                                                          std::int64_t global_n,
                                                          const std::vector<Triplet>& t,
                                                          std::shared_ptr<const Reducer> reducer,
                                                          const RowOwnership& owners) {
  FUSE_ASSERT(row_lo <= row_hi, "DistSparseMatrix::from_triplets: row_lo {} > row_hi {}", row_lo,
             row_hi);
  FUSE_ASSERT(!owners.row_lo.empty() && owners.row_lo.front() == 0,
             "RowOwnership::row_lo must start at 0 and name every participant; got {} entries",
             owners.row_lo.size());
  for (std::size_t r = 0; r + 1 < owners.row_lo.size(); ++r) {
    FUSE_ASSERT(owners.row_lo[r] < owners.row_lo[r + 1],
               "RowOwnership::row_lo must be strictly ascending; row_lo[{}]={} >= row_lo[{}]={}",
               r, owners.row_lo[r], r + 1, owners.row_lo[r + 1]);
  }
  FUSE_ASSERT(reducer != nullptr, "DistSparseMatrix::from_triplets built with a null Reducer");

  const auto local_rows = static_cast<std::size_t>(row_hi - row_lo);

  std::vector<LocalEntry> diag_entries;
  std::vector<LocalEntry> off_entries;  // col temporarily holds the GLOBAL column
  diag_entries.reserve(t.size());
  off_entries.reserve(t.size());

  for (const Triplet& e : t) {
    if (e.row < row_lo || e.row >= row_hi) {
      return Failure{"E-LINALG-FOREIGN-ROW",
                    fmt::format("triplet row {} is outside this participant's own range "
                               "[{}, {}); from_triplets takes only this participant's rows",
                               e.row, row_lo, row_hi)};
    }
    if (e.col < 0 || e.col >= global_n) {
      return Failure{"E-LINALG-COL-RANGE",
                    fmt::format("triplet (row {}, col {}) names a column outside [0, {})", e.row,
                               e.col, global_n)};
    }
    const auto local_row = static_cast<std::size_t>(e.row - row_lo);
    if (e.col >= row_lo && e.col < row_hi) {
      diag_entries.push_back({local_row, static_cast<std::size_t>(e.col - row_lo), e.value});
    } else {
      off_entries.push_back({local_row, static_cast<std::size_t>(e.col), e.value});
    }
  }

  // Unique off-process columns, ascending -- which also groups them by
  // owner ascending, since RowOwnership's ranges are contiguous and
  // monotonic (the same reasoning the header comment above gives for
  // owner_of's single upper_bound).
  std::map<std::size_t, std::size_t> ghost_index_of;  // global col -> ghost local index
  {
    std::vector<std::size_t> unique_cols;
    unique_cols.reserve(off_entries.size());
    for (const LocalEntry& e : off_entries) unique_cols.push_back(e.col);
    std::sort(unique_cols.begin(), unique_cols.end());
    unique_cols.erase(std::unique(unique_cols.begin(), unique_cols.end()), unique_cols.end());
    for (std::size_t g = 0; g < unique_cols.size(); ++g) ghost_index_of[unique_cols[g]] = g;
  }

  GhostPattern pattern;
  pattern.owner_rank.resize(ghost_index_of.size());
  pattern.remote_local_index.resize(ghost_index_of.size());
  for (const auto& [global_col, g] : ghost_index_of) {
    const int owner = owner_of(static_cast<std::int64_t>(global_col), owners);
    pattern.owner_rank[g] = owner;
    pattern.remote_local_index[g] =
        static_cast<std::int64_t>(global_col) - owners.row_lo[static_cast<std::size_t>(owner)];
  }

  // off_entries.col is still the GLOBAL column; rewrite in place to the
  // ghost-local index the GhostPattern above just assigned, now that every
  // off-process column has one.
  for (LocalEntry& e : off_entries) e.col = ghost_index_of.at(e.col);

  DistSparseMatrix m;
  m.row_lo_ = row_lo;
  m.global_n_ = global_n;
  m.local_rows_ = local_rows;
  m.reducer_ = std::move(reducer);
  m.ghost_pattern_ = std::move(pattern);
  assemble_csr(diag_entries, local_rows, m.diag_row_ptr_, m.diag_col_idx_, m.diag_val_);
  assemble_csr(off_entries, local_rows, m.off_row_ptr_, m.off_col_idx_, m.off_val_);
  return m;
}

Result<void> DistSparseMatrix::apply(DistVector& x, DistVector& y) const {
  FUSE_ASSERT(x.owned_size() == local_rows_,
             "DistSparseMatrix::apply: x owns {} entries, this matrix owns {} rows", x.owned_size(),
             local_rows_);
  FUSE_ASSERT(y.owned_size() == local_rows_,
             "DistSparseMatrix::apply: y owns {} entries, this matrix owns {} rows", y.owned_size(),
             local_rows_);
  FUSE_ASSERT(x.ghost_size() == ghost_pattern_.owner_rank.size(),
             "DistSparseMatrix::apply: x carries {} ghosts, this matrix's ghost_pattern() "
             "carries {}; x was not built from this matrix's own ghost_pattern()",
             x.ghost_size(), ghost_pattern_.owner_rank.size());
  // Row i's write to y overwrites x's owned
  // block before every later row has read it as an off-diagonal input
  // column, so apply(v, v) (or any partial overlap) silently computes
  // the wrong answer -- caught here, not left to the header's own
  // no-aliasing sentence alone. Range comparison via std::less, the same
  // total-order argument src/num/accel_relaxation.cpp's own overlap
  // check already relies on.
  {
    const std::less<const double*> before;
    const auto overlaps = [&before](const double* a, const double* b, std::size_t n) {
      return before(a, b + n) && before(b, a + n);
    };
    FUSE_ASSERT(!overlaps(x.data(), y.data(), x.owned_size()),
               "DistSparseMatrix::apply: x and y must not alias, not even partially");
  }

  if (auto r = x.update_ghosts(); !r.ok()) return r.error();

  const double* xd = x.data();
  double* yd = y.data();
  for (std::size_t i = 0; i < local_rows_; ++i) {
    double acc = 0.0;
    for (std::size_t k = diag_row_ptr_[i]; k < diag_row_ptr_[i + 1]; ++k) {
      acc += diag_val_[k] * xd[diag_col_idx_[k]];
    }
    for (std::size_t k = off_row_ptr_[i]; k < off_row_ptr_[i + 1]; ++k) {
      acc += off_val_[k] * xd[local_rows_ + off_col_idx_[k]];
    }
    yd[i] = acc;
  }
  return Result<void>::success();
}

void DistSparseMatrix::diagonal(DistVector& d) const {
  FUSE_ASSERT(d.owned_size() == local_rows_,
             "DistSparseMatrix::diagonal: d owns {} entries, this matrix owns {} rows",
             d.owned_size(), local_rows_);
  double* dd = d.data();
  for (std::size_t i = 0; i < local_rows_; ++i) {
    const auto begin = diag_col_idx_.begin() + static_cast<std::ptrdiff_t>(diag_row_ptr_[i]);
    const auto end = diag_col_idx_.begin() + static_cast<std::ptrdiff_t>(diag_row_ptr_[i + 1]);
    const auto it = std::lower_bound(begin, end, i);
    dd[i] = (it != end && *it == i)
               ? diag_val_[static_cast<std::size_t>(std::distance(diag_col_idx_.begin(), it))]
               : 0.0;
  }
}

}  // namespace fuse::linalg
