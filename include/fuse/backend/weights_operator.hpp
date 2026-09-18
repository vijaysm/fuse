#pragma once

/// `RemapWeights` as a `LinearOperator`.
///
/// This is the adapter that makes the remap path -- the always-available
/// conservative one built from an exact simplex supermesh, and later the MOAB
/// and TempestRemap tiers, which produce the same triplet structure -- an
/// operator like any other. Nothing about it is remap-specific beyond the
/// source of the triplets, so a persisted map reloaded from disk (S5.4) and a
/// map built in memory give the same operator.
///
/// It does NOT copy the weights. The plan holds one of these per remap for the
/// whole run, and the triplet arrays are the largest thing in a coupled setup
/// after the fields themselves.

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

#include "fuse/backend/linear_operator.hpp"
#include "fuse/backend/remap_plan.hpp"

namespace fuse {

class WeightsOperator final : public LinearOperator {
 public:
  /// `weights` must outlive the operator. Asserted rather than trusted: the
  /// constructor is the only place that could check, and a dangling weights
  /// reference produces plausible numbers rather than a crash.
  explicit WeightsOperator(const RemapWeights& weights) noexcept;

  [[nodiscard]] std::size_t rows() const noexcept override;
  [[nodiscard]] std::size_t cols() const noexcept override;
  [[nodiscard]] std::string_view kind() const noexcept override { return "remap-weights"; }

  /// Read from the weights' own audit rather than asserted here: the builder
  /// measured the row and column defects when it built them, so claiming
  /// conservation is a statement about THIS matrix and not about the method
  /// that produced it. A map reloaded from disk carries the same audit.
  [[nodiscard]] OperatorClaims claims() const noexcept override;

  [[nodiscard]] Result<void> apply(std::span<const double> x, std::span<double> y,
                                   int components) const override;
  [[nodiscard]] Result<void> apply_transpose(std::span<const double> y,
                                             std::span<double> x,
                                             int components) const override;
  [[nodiscard]] Result<void> row_sums(std::span<double> out) const override;
  [[nodiscard]] Result<void> col_sums(std::span<double> out) const override;
  [[nodiscard]] std::uint64_t signature() const override;

 private:
  const RemapWeights* w_;
};

}  // namespace fuse
