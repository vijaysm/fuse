#include "fuse/backend/weights_operator.hpp"

#include <cstddef>
#include <bit>
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>

#include "fuse/base/bytes.hpp"
#include "fuse/base/error.hpp"

namespace fuse {

namespace {

/// The defect below which the builder's audit is treated as satisfied. Not a
/// tolerance on the physics: the builder already compares against its own,
/// tighter bound and records what it measured. This one only decides whether to
/// CLAIM the property, and a matrix whose worst column defect is 1e-12 is
/// conservative for every purpose the claim is used for (an init-time check and
/// the composition rule).
constexpr double kClaimTolerance = 1e-10;

}  // namespace

WeightsOperator::WeightsOperator(const RemapWeights& weights) noexcept
    : w_(&weights) {}

std::size_t WeightsOperator::rows() const noexcept { return w_->n_rows; }
std::size_t WeightsOperator::cols() const noexcept { return w_->n_cols; }

OperatorClaims WeightsOperator::claims() const noexcept {
  return OperatorClaims{
      .preserves_constants = w_->max_row_defect <= kClaimTolerance,
      // REVIEW.md TF-16: `max_col_defect` alone is not enough -- it is only
      // a conservation residual, the property this flag promises, when the
      // builder that produced it says so (RemapWeights::
      // col_defect_is_measure_weighted; false for the consistent-
      // interpolation (H) builders, whose column sum has no measure behind
      // it and can coincidentally read near 1 without being conservative).
      .conservative = w_->col_defect_is_measure_weighted &&
                      w_->max_col_defect <= kClaimTolerance,
      .monotone = w_->monotone,
      // A weights matrix is expressed in local indices on one rank and touches
      // nothing remote; the halo exchange that may precede it is a separate
      // operator, which is the reason they are separate operators.
      .rank_local = true,
  };
}

Result<void> WeightsOperator::apply(std::span<const double> x,
                                    std::span<double> y, int components) const {
  if (components < 1) {
    return Failure{"E-OP-ARG", "WeightsOperator::apply: components must be >= 1"};
  }
  const std::size_t nc = static_cast<std::size_t>(components);
  if (x.size() != w_->n_cols * nc || y.size() != w_->n_rows * nc) {
    return Failure{"E-OP-SHAPE",
                   "WeightsOperator::apply: got " + std::to_string(x.size()) +
                       " in and " + std::to_string(y.size()) + " out for a " +
                       std::to_string(w_->n_rows) + "x" +
                       std::to_string(w_->n_cols) + " operator with " +
                       std::to_string(components) + " components"};
  }
  // Delegates: apply_weights already validates the triplets against the shape
  // and is the path the runtime has been using, so there is exactly one sparse
  // multiply in the library rather than a second one that could drift from it.
  return apply_weights(*w_, x.data(), y.data(), components);
}

Result<void> WeightsOperator::apply_transpose(std::span<const double> y,
                                              std::span<double> x,
                                              int components) const {
  if (components < 1) {
    return Failure{"E-OP-ARG",
                   "WeightsOperator::apply_transpose: components must be >= 1"};
  }
  const std::size_t nc = static_cast<std::size_t>(components);
  if (y.size() != w_->n_rows * nc || x.size() != w_->n_cols * nc) {
    return Failure{"E-OP-SHAPE",
                   "WeightsOperator::apply_transpose: got " +
                       std::to_string(y.size()) + " in and " +
                       std::to_string(x.size()) + " out for a " +
                       std::to_string(w_->n_rows) + "x" +
                       std::to_string(w_->n_cols) + " operator with " +
                       std::to_string(components) + " components"};
  }
  if (w_->row.size() != w_->val.size() || w_->col.size() != w_->val.size()) {
    return Failure{"E-REMAP-COO",
                   "WeightsOperator::apply_transpose: the triplet arrays "
                   "disagree in length"};
  }
  for (double& v : x) v = 0.0;
  // The same traversal as apply with the two indices exchanged, deliberately
  // written as one loop over the SAME triplets rather than by transposing the
  // matrix: a transposed copy is a second thing to keep in step, and the
  // adjoint identity would then be testing the copy rather than the operator.
  for (std::size_t k = 0; k < w_->val.size(); ++k) {
    const std::int32_t r = w_->row[k];
    const std::int32_t c = w_->col[k];
    if (r < 0 || static_cast<std::size_t>(r) >= w_->n_rows || c < 0 ||
        static_cast<std::size_t>(c) >= w_->n_cols) {
      return Failure{"E-REMAP-COO",
                     "WeightsOperator::apply_transpose: triplet " +
                         std::to_string(k) + " is outside the matrix"};
    }
    const std::size_t ro = static_cast<std::size_t>(r) * nc;
    const std::size_t co = static_cast<std::size_t>(c) * nc;
    for (std::size_t comp = 0; comp < nc; ++comp) {
      x[co + comp] += w_->val[k] * y[ro + comp];
    }
  }
  return Result<void>::success();
}

Result<void> WeightsOperator::row_sums(std::span<double> out) const {
  if (out.size() != w_->n_rows) {
    return Failure{"E-OP-SHAPE", "WeightsOperator::row_sums: expected " +
                                     std::to_string(w_->n_rows) + " entries, got " +
                                     std::to_string(out.size())};
  }
  for (double& v : out) v = 0.0;
  for (std::size_t k = 0; k < w_->val.size(); ++k) {
    const std::int32_t r = w_->row[k];
    if (r < 0 || static_cast<std::size_t>(r) >= w_->n_rows) {
      return Failure{"E-REMAP-COO", "WeightsOperator::row_sums: triplet " +
                                        std::to_string(k) + " has row " +
                                        std::to_string(r)};
    }
    out[static_cast<std::size_t>(r)] += w_->val[k];
  }
  return Result<void>::success();
}

Result<void> WeightsOperator::col_sums(std::span<double> out) const {
  if (out.size() != w_->n_cols) {
    return Failure{"E-OP-SHAPE", "WeightsOperator::col_sums: expected " +
                                     std::to_string(w_->n_cols) + " entries, got " +
                                     std::to_string(out.size())};
  }
  for (double& v : out) v = 0.0;
  for (std::size_t k = 0; k < w_->val.size(); ++k) {
    const std::int32_t c = w_->col[k];
    if (c < 0 || static_cast<std::size_t>(c) >= w_->n_cols) {
      return Failure{"E-REMAP-COO", "WeightsOperator::col_sums: triplet " +
                                        std::to_string(k) + " has column " +
                                        std::to_string(c)};
    }
    out[static_cast<std::size_t>(c)] += w_->val[k];
  }
  return Result<void>::success();
}

std::uint64_t WeightsOperator::signature() const {
  // Over the shape and the triplets only. Explicitly NOT over the audit fields
  // (coverage, defects, measures): those are measurements ABOUT this matrix, so
  // including them would make a map that computes exactly the same thing look
  // like a different operator if the builder's diagnostics ever changed. The
  // order of the triplets IS included, because it is fixed (row-ascending) and
  // because two orders that differ give different rounding.
  std::vector<std::byte> out;
  out.reserve(32 + w_->val.size() * 16);
  bytes::put_integer(out, static_cast<std::uint64_t>(w_->n_rows), 8);
  bytes::put_integer(out, static_cast<std::uint64_t>(w_->n_cols), 8);
  bytes::put_integer(out, static_cast<std::uint64_t>(w_->val.size()), 8);
  bytes::put_signed(out, w_->order, 4);
  for (std::size_t k = 0; k < w_->val.size(); ++k) {
    bytes::put_signed(out, w_->row[k], 4);
    bytes::put_signed(out, w_->col[k], 4);
    // The bit pattern, not a formatted value: two weights that differ in the
    // last bit compute different results, and a decimal rendering would hash
    // them the same.
    bytes::put_integer(out, std::bit_cast<std::uint64_t>(w_->val[k]), 8);
  }
  return bytes::fnv1a(out);
}

}  // namespace fuse
