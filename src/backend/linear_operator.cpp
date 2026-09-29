#include "fuse/backend/linear_operator.hpp"

#include <cmath>
#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace fuse {

namespace {

/// Straight dot product. Not a Kokkos reduction: these are init-time checks on
/// vectors the caller already holds in host memory, and a reduction here would
/// associate differently from the two applies it is comparing, which is the
/// one thing this must not do.
[[nodiscard]] double dot(std::span<const double> a, std::span<const double> b) {
  double s = 0.0;
  const std::size_t n = a.size() < b.size() ? a.size() : b.size();
  for (std::size_t i = 0; i < n; ++i) s += a[i] * b[i];
  return s;
}

}  // namespace

Result<double> adjoint_defect(const LinearOperator& a,
                              std::span<const double> x,
                              std::span<const double> y,
                              int components) {
  if (components < 1) {
    return Failure{"E-OP-ARG", "adjoint_defect: components must be >= 1, not " +
                                   std::to_string(components)};
  }
  const std::size_t nc = static_cast<std::size_t>(components);
  if (x.size() != a.cols() * nc || y.size() != a.rows() * nc) {
    return Failure{"E-OP-SHAPE",
                   "adjoint_defect: x has " + std::to_string(x.size()) +
                       " values and y has " + std::to_string(y.size()) +
                       ", but the operator is " + std::to_string(a.rows()) +
                       "x" + std::to_string(a.cols()) + " with " +
                       std::to_string(components) + " components"};
  }

  std::vector<double> ax(a.rows() * nc, 0.0);
  std::vector<double> aty(a.cols() * nc, 0.0);
  auto r = a.apply(x, ax, components);
  if (!r.ok()) return r.error();
  r = a.apply_transpose(y, aty, components);
  if (!r.ok()) return r.error();

  const double lhs = dot(y, ax);   // y . A x
  const double rhs = dot(aty, x);  // A^T y . x
  // Relative to the magnitude of the terms, because the absolute difference of
  // two large dot products says nothing on its own. The denominator is floored
  // at 1 so an operator that maps everything to zero reports 0 rather than a
  // division by zero.
  const double scale = std::abs(lhs) + std::abs(rhs);
  return std::abs(lhs - rhs) / (scale > 1.0 ? scale : 1.0);
}

namespace {

/// A o B. Holds its factors by shared_ptr because a composed operator commonly
/// shares a factor with another composition -- the same remap appearing in two
/// field transfers -- and copying weight arrays per composition would be the largest
/// avoidable allocation in the init path.
class ComposedOperator final : public LinearOperator {
 public:
  ComposedOperator(std::shared_ptr<const LinearOperator> outer,
                   std::shared_ptr<const LinearOperator> inner)
      : outer_(std::move(outer)),
        inner_(std::move(inner)),
        kind_(std::string(outer_->kind()) + "∘" + std::string(inner_->kind())) {}

  [[nodiscard]] std::size_t rows() const noexcept override { return outer_->rows(); }
  [[nodiscard]] std::size_t cols() const noexcept override { return inner_->cols(); }
  [[nodiscard]] std::string_view kind() const noexcept override { return kind_; }

  /// A claim survives composition only if BOTH factors make it -- and for
  /// conservation that is the mathematically right rule, since composing a
  /// conservative map with a non-conservative one is not conservative.
  /// Monotonicity is the same. This is why the fixture checks claims rather
  /// than a fixed list: the composed operator's claims are computed, so a
  /// composition cannot silently inherit a property it does not have.
  [[nodiscard]] OperatorClaims claims() const noexcept override {
    const OperatorClaims a = outer_->claims();
    const OperatorClaims b = inner_->claims();
    return OperatorClaims{
        .preserves_constants = a.preserves_constants && b.preserves_constants,
        .conservative = a.conservative && b.conservative,
        .monotone = a.monotone && b.monotone,
        .rank_local = a.rank_local && b.rank_local,
    };
  }

  [[nodiscard]] Result<void> apply(std::span<const double> x, std::span<double> y,
                                   int components) const override {
    if (components < 1) {
      return Failure{"E-OP-ARG", "compose: components must be >= 1"};
    }
    const std::size_t nc = static_cast<std::size_t>(components);
    std::vector<double> mid(inner_->rows() * nc, 0.0);
    auto r = inner_->apply(x, mid, components);
    if (!r.ok()) return r;
    return outer_->apply(mid, y, components);
  }

  /// (A B)^T = B^T A^T. Getting this backwards is the classic composition bug
  /// and it is invisible on a symmetric factor, which is exactly why the
  /// fixture composes two DIFFERENT non-symmetric operators.
  [[nodiscard]] Result<void> apply_transpose(std::span<const double> y,
                                             std::span<double> x,
                                             int components) const override {
    if (components < 1) {
      return Failure{"E-OP-ARG", "compose: components must be >= 1"};
    }
    const std::size_t nc = static_cast<std::size_t>(components);
    std::vector<double> mid(outer_->cols() * nc, 0.0);
    auto r = outer_->apply_transpose(y, mid, components);
    if (!r.ok()) return r;
    return inner_->apply_transpose(mid, x, components);
  }

  /// Row sums of A B are A applied to B's row sums, and symmetrically for
  /// columns -- both because a sum over a row of the product is a sum over rows
  /// of the factors, which is what applying to a vector of ones computes.
  [[nodiscard]] Result<void> row_sums(std::span<double> out) const override {
    std::vector<double> inner_rows(inner_->rows(), 0.0);
    auto r = inner_->row_sums(inner_rows);
    if (!r.ok()) return r;
    return outer_->apply(inner_rows, out, 1);
  }
  [[nodiscard]] Result<void> col_sums(std::span<double> out) const override {
    std::vector<double> outer_cols(outer_->cols(), 0.0);
    auto r = outer_->col_sums(outer_cols);
    if (!r.ok()) return r;
    return inner_->apply_transpose(outer_cols, out, 1);
  }

  /// Order-dependent by construction: A∘B and B∘A are different operators, so
  /// mixing the factor signatures commutatively would let a plan reuse the
  /// wrong persisted map. The constant is arbitrary but fixed; it is written
  /// into files.
  [[nodiscard]] std::uint64_t signature() const override {
    constexpr std::uint64_t kComposeTag = 0x9E3779B97F4A7C15ULL;
    std::uint64_t h = kComposeTag;
    h = (h ^ outer_->signature()) * 0x100000001B3ULL;
    h = (h ^ inner_->signature()) * 0x100000001B3ULL;
    return h;
  }

 private:
  std::shared_ptr<const LinearOperator> outer_;
  std::shared_ptr<const LinearOperator> inner_;
  std::string kind_;
};

}  // namespace

Result<std::unique_ptr<LinearOperator>> compose(
    std::shared_ptr<const LinearOperator> outer,
    std::shared_ptr<const LinearOperator> inner) {
  if (!outer || !inner) {
    return Failure{"E-OP-ARG", "compose: null operator"};
  }
  if (outer->cols() != inner->rows()) {
    return Failure{"E-OP-SHAPE",
                   "compose: the outer operator takes " +
                       std::to_string(outer->cols()) +
                       " values but the inner one produces " +
                       std::to_string(inner->rows())};
  }
  return std::unique_ptr<LinearOperator>{
      new ComposedOperator(std::move(outer), std::move(inner))};
}

}  // namespace fuse
