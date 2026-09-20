#pragma once

/// Iteration configuration: WHAT was asked for, not how it is computed.
///
/// Moved down from the numerics layer during the merge. The graph's iteration
/// region has to declare a convergence criterion and an acceleration choice, and a
/// model layer that reached up into numerics to do so inverted the dependency (the
/// layering gate caught it). The algorithms -- the `Accelerator` implementations and
/// the criterion's per-iteration evaluation -- stay in `fuse::num`, which depends on
/// this header rather than the reverse.

#include <cmath>
#include <string>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "fuse/base/error.hpp"
#include "fuse/base/result.hpp"

namespace fuse {

enum class MeasureKind {
  /// |r| / |x| <= tol, with |x| the reference supplied by the caller
  /// (typically the norm of the current coupling value).
  Relative,
  /// |r| <= tol. The reference is ignored.
  Absolute,
  /// Either of the two above.
  AbsoluteOrRelative,
  /// Same arithmetic as Relative; the distinction is *what the caller passes
  /// as the reference*. Relative divides by the norm of the current value,
  /// ResidualRelative by the norm of the first residual of the current window,
  /// i.e. it measures the reduction of the residual rather than its size
  /// relative to the solution. The enumerator exists so the plan dump and the
  /// report can say which of the two a measure means; the comparison itself is
  /// identical by construction.
  ResidualRelative,
};

[[nodiscard]] constexpr const char* to_cstring(MeasureKind k) noexcept {
  switch (k) {
    case MeasureKind::Relative:           return "Relative";
    case MeasureKind::Absolute:           return "Absolute";
    case MeasureKind::AbsoluteOrRelative: return "AbsoluteOrRelative";
    case MeasureKind::ResidualRelative:   return "ResidualRelative";
  }
  return "?";
}

namespace detail {

/// std::to_string(double) and a default-constructed ostringstream both format
/// through the global locale, so their output changes with LC_NUMERIC (0.5
/// becomes "0,5" under de_DE). The convergence report is diagnostic text that
/// gets logged, diffed and matched, so it is formatted through the classic "C"
/// locale instead. Same reasoning as detail::format_double in ir/dump.hpp; the
/// name differs so both headers can be included in one translation unit.
[[nodiscard]] inline std::string format_convergence_double(double v) {
  // Was an ostringstream imbued with the classic locale, to stop a
  // comma-decimal-separator locale from changing the text of a diagnostic.
  // fmt is locale-independent unless asked otherwise, so the imbue is gone and
  // the property it protected still holds.
  return fmt::format("{}", v);
}

}  // namespace detail

struct ConvergenceMeasure {
  MeasureKind kind = MeasureKind::Relative;
  double      tolerance = 1e-6;
  std::string field;

  /// True when this measure alone considers the iteration converged.
  ///
  /// `reference_norm` is ignored by Absolute and used by every other kind.
  /// REVIEW.md R-34: this used to point at "the header comment" for the
  /// zero-reference, non-finite and AND/OR policies -- no such section
  /// exists in this file. They are implemented, and explained, inline
  /// below: a zero-or-negative `reference_norm` falls back to the absolute
  /// check (`have_ref`); a non-finite norm reports "not converged" rather
  /// than throwing; and `AbsoluteOrRelative` is the OR of the two (there is
  /// no AND case among the current MeasureKind values). Throws
  /// std::invalid_argument on a non-positive or non-finite
  /// tolerance, or on a negative (finite) norm -- neither can arise from
  /// fuse::num::norm_l2 and friends, so both indicate a caller bug that must
  /// not be swallowed in a release build.
  [[nodiscard]] bool satisfied(double residual_norm,
                               double reference_norm) const {
    if (!(tolerance > 0.0) || !std::isfinite(tolerance)) {
      FUSE_THROW(InvalidArgument, "{}", std::string() + "ConvergenceMeasure::satisfied: tolerance must be finite and "
          "positive, got " +
          detail::format_convergence_double(tolerance) + " for field '" +
          field + "'");
    }
    // Divergence, not a bug: never converged.
    if (!std::isfinite(residual_norm) || !std::isfinite(reference_norm)) {
      return false;
    }
    if (residual_norm < 0.0 || reference_norm < 0.0) {
      FUSE_THROW(InvalidArgument, "{}", std::string() + "ConvergenceMeasure::satisfied: norms must be non-negative, got "
          "residual " +
          detail::format_convergence_double(residual_norm) + " and reference " +
          detail::format_convergence_double(reference_norm) + " for field '" +
          field + "'");
    }

    const bool absolute_ok = residual_norm <= tolerance;
    const bool have_ref = reference_norm > 0.0;
    // reference_norm > 0 and both operands finite, so the quotient is
    // well-defined; it can overflow to +inf for a denormal reference, and
    // inf <= tol is false, i.e. the overflow errs towards "not converged".
    const bool relative_ok =
        have_ref ? (residual_norm / reference_norm <= tolerance) : absolute_ok;

    switch (kind) {
      case MeasureKind::Absolute:           return absolute_ok;
      case MeasureKind::Relative:
      case MeasureKind::ResidualRelative:   return relative_ok;
      case MeasureKind::AbsoluteOrRelative: return absolute_ok || relative_ok;
    }
    // Unreachable for a valid enumerator. An out-of-range cast reports "not
    // converged", which costs an iteration rather than accepting a wrong
    // answer.
    return false;
  }

  /// The quantity the measure actually compares against `tolerance`, so a
  /// report can print the number the decision was made on. Mirrors
  /// satisfied() exactly, including the zero-reference fallback.
  [[nodiscard]] double compared_value(double residual_norm,
                                      double reference_norm) const {
    if (kind == MeasureKind::Absolute) return residual_norm;
    if (!std::isfinite(residual_norm) || !std::isfinite(reference_norm)) {
      return residual_norm;
    }
    if (!(reference_norm > 0.0)) return residual_norm;
    const double ratio = residual_norm / reference_norm;
    // AbsoluteOrRelative passes if *either* holds, so report whichever branch
    // is closer to passing.
    if (kind == MeasureKind::AbsoluteOrRelative) {
      return ratio < residual_norm ? ratio : residual_norm;
    }
    return ratio;
  }
};

/// Conjunction of measures: `Relative(...) & Absolute(...)` from spec §5.3.
///
/// Measure i is tested against `residual_norms[i]` and `reference_norms[i]`,
/// so the caller's ordering is load-bearing; both vectors must have exactly
/// size() entries or the call throws.
///
/// An empty criterion is satisfied -- true is the identity of AND, and
/// all_satisfied() has nothing to refuse. It is *not* a safe default for an
/// iteration region: a region with no measures would stop after one iteration.
/// Rejecting that is the region's job, and empty() is provided for it.
class ConvergenceCriterion {
 public:
  void add(ConvergenceMeasure m) { measures_.push_back(std::move(m)); }

  [[nodiscard]] std::size_t size() const noexcept { return measures_.size(); }
  [[nodiscard]] bool empty() const noexcept { return measures_.empty(); }
  [[nodiscard]] const std::vector<ConvergenceMeasure>& measures()
      const noexcept {
    return measures_;
  }

  /// Every measure must hold. Throws std::invalid_argument if either input
  /// does not have exactly size() entries: padding a short vector with zeros
  /// would report a missing residual as converged, which is the one failure
  /// mode this class exists to prevent, and it must be caught under -DNDEBUG
  /// too.
  [[nodiscard]] bool all_satisfied(
      const std::vector<double>& residual_norms,
      const std::vector<double>& reference_norms) const {
    check_sizes("all_satisfied", residual_norms, reference_norms);
    for (std::size_t i = 0; i < measures_.size(); ++i) {
      if (!measures_[i].satisfied(residual_norms[i], reference_norms[i])) {
        return false;
      }
    }
    return true;
  }

  /// One human-readable line naming every field, the value compared, the
  /// tolerance and the per-measure state. Formatted in the classic locale so
  /// the text does not depend on LC_NUMERIC. Same size contract as
  /// all_satisfied().
  [[nodiscard]] std::string report(
      const std::vector<double>& residual_norms,
      const std::vector<double>& reference_norms) const {
    check_sizes("report", residual_norms, reference_norms);
    if (measures_.empty()) return "(no measures)";
    std::string out;
    for (std::size_t i = 0; i < measures_.size(); ++i) {
      const ConvergenceMeasure& m = measures_[i];
      const double r = residual_norms[i];
      const double ref = reference_norms[i];
      if (i != 0) out += ", ";
      out += m.field;
      out += '[';
      out += to_cstring(m.kind);
      out += "] ";
      out += detail::format_convergence_double(m.compared_value(r, ref));
      out += " vs tol ";
      out += detail::format_convergence_double(m.tolerance);
      out += m.satisfied(r, ref) ? " [ok]" : " [pending]";
    }
    return out;
  }

 private:
  void check_sizes(const char* fn, const std::vector<double>& residual_norms,
                   const std::vector<double>& reference_norms) const {
    if (residual_norms.size() != measures_.size() ||
        reference_norms.size() != measures_.size()) {
      FUSE_THROW(InvalidArgument, "{}", std::string() + std::string{"ConvergenceCriterion::"} + fn + ": expected " +
          std::to_string(measures_.size()) +
          " residual and reference norms, got " +
          std::to_string(residual_norms.size()) + " and " +
          std::to_string(reference_norms.size()));
    }
  }

  std::vector<ConvergenceMeasure> measures_;
};

namespace detail {

/// Validating factory shared by the named constructors in fuse::Converged. The same check
/// runs again in satisfied(), because ConvergenceMeasure is an aggregate and
/// can be brace-initialised past this function.

[[nodiscard]] inline ConvergenceMeasure make_measure(MeasureKind kind,
                                                     double tol,
                                                     std::string field) {
  if (!(tol > 0.0) || !std::isfinite(tol)) {
    FUSE_THROW(InvalidArgument, "{}", std::string() + std::string{"Converged::"} + to_cstring(kind) +
        ": tolerance must be finite and positive, got " +
        format_convergence_double(tol) + " for field '" + field + "'");
  }
  return ConvergenceMeasure{kind, tol, std::move(field)};
}

}  // namespace detail

enum class AccelKind {
  None,
  Constant,
  Aitken,
  Secant,
  Steffensen,
  IQN_ILS,
  IQN_IMVJ,
};

/// Column filters for the quasi-Newton least squares, as defined by preCICE
/// and by Davis, Schulte & Uekermann, Math. Comput. Appl. 27(3):40 (2022).
/// With |R_jj| the length of the part of column j that the newer columns do
/// not already explain, and `limit` = AccelConfig::limit:
///   None  keep every column.
///   QR1   drop column j if |R_jj| < limit * ||V||_F. One threshold for the
///         whole matrix, so it wants a much smaller limit than QR2 (preCICE
///         suggests 1e-5..1e-6). It can drop every column, including the
///         newest.
///   QR2   drop column j if |R_jj| < limit * ||v_j||. A per-column threshold on
///         the fraction of the column that is new, invariant under rescaling
///         V. The newest column is always kept, so a non-empty history stays
///         non-empty.
///   QR3   the same retained column set as QR2. In preCICE, QR3 is a cost
///         optimisation that skips QR2's en-bloc refactorisation when a cheap
///         test on the incrementally maintained QR says nothing would be
///         dropped; this implementation keeps no factorisation between calls,
///         so the test and the filter read the same matrix and the outcomes
///         coincide exactly. See `fuse::detail::filter_qn_columns`.
/// See `AccelConfig::filter` and the seam documented there.
enum class FilterKind { None, QR1, QR2, QR3 };

[[nodiscard]] const char* to_cstring(AccelKind k) noexcept;
[[nodiscard]] const char* to_cstring(FilterKind f) noexcept;

struct AccelConfig {
  AccelKind kind = AccelKind::Constant;

  /// Relaxation factor used whenever no usable history exists -- iteration 0
  /// of a window for every method, and any later iteration whose secant
  /// information is degenerate. Must be finite and non-zero.
  double initial_relaxation = 0.1;

  /// Fixed factor for AccelKind::Constant. Must be finite and non-zero.
  double omega = 0.5;

  /// Maximum number of secant column pairs retained within one window. It
  /// also caps the number of past windows IQN_IMVJ chains, at
  /// min(reuse, Accelerator::kMaxJacobianBlocks). Must be in
  /// [1, Accelerator::kMaxReuse]. AccelKind::Secant ignores it and always
  /// keeps exactly one pair.
  ///
  /// MEASURED: raising this above 8 is a NO-OP whenever a filter is active.
  /// Task 7.6's Dirichlet-Neumann demo ran reuse = 8/16/24/32 with QR1 and with
  /// QR2 and got byte-identical sweep counts in every case -- the filter drops
  /// columns long before the history cap binds. Tune `filter`/`limit` first;
  /// `reuse` only bites with FilterKind::None.
  int reuse = 8;

  /// Column filter applied before the least-squares solve.
  ///
  /// `fuse::detail::filter_qn_columns` in src/num/accel_quasi_newton.cpp is the
  /// single point where a filter drops columns, and the definitions and their
  /// sources are documented there. Dropped columns are removed from the stored
  /// history for good, not merely skipped for one solve, so `columns()` reports
  /// what survived.
  ///
  /// A filter is NOT the mechanism that keeps the solve stable: rank deficiency
  /// is handled unconditionally by the rank-revealing pivoted QR in the solver,
  /// which truncates at a machine-precision threshold regardless of this
  /// setting. A filter can only make the retained set smaller than what that
  /// truncation would already accept.
  ///
  /// Filtering is not free on well-conditioned problems: it discards secant
  /// information that a clean least-squares fit could still have used. THE
  /// DEFAULT IS PRECICE'S AND IS TUNED FOR ILL-CONDITIONED FSI. Two independent
  /// measurements in this tree say what it costs when the problem is not:
  ///   * Accelerator.LadderIsMonotoneOnADenseLinearProblem, an exactly-linear
  ///     5-dof problem: IQN_ILS and IQN_IMVJ each take 7 iterations at the
  ///     default and 6 with FilterKind::None. The oldest column's orthogonal
  ///     fraction crosses the 1e-2 limit at iteration 4 and is dropped -- but
  ///     on an exactly-linear problem that 0.18% IS the last missing
  ///     direction, so the exact rank-5 fit lands one iteration later. The
  ///     test pins the count and explains the mechanism.
  ///   * examples/dn_heat, 32 interface stations x 4 conductivity strips over
  ///     20 periods: **211 sweeps unfiltered, 4233 with the default -- a factor
  ///     of 20 -- and one window reaches max_iterations without converging.**
  ///     QR1 at the same limit is worse still (6638-10935 sweeps).
  ///
  /// The default is left at preCICE's, because that is the reference
  /// implementation and diverging from it silently would be its own trap. But
  /// if your problem is well conditioned, FilterKind::None is very likely the
  /// right choice, and a window that will not converge under the default is a
  /// reason to try it before suspecting the accelerator.
  FilterKind filter = FilterKind::QR2;

  /// Threshold consumed by the filter, in the units each filter defines (see
  /// FilterKind). Validated here as finite and in [0, 1); 0 disables the filter
  /// exactly as FilterKind::None does. 1e-2 is preCICE's default for the
  /// relative filters QR2 and QR3; QR1 wants something nearer 1e-5.
  double limit = 1e-2;

  /// Scale each field block of the residual by the reciprocal of its own
  /// 2-norm before the least-squares solve, so a large-magnitude field cannot
  /// dominate the fit. Requires set_field_partition() to have any effect, for
  /// two separate reasons:
  ///   * the default partition is *empty*, not one block. `field_weights`
  ///     returns exactly 1.0 for every entry in that case and no reciprocal
  ///     norm is ever formed, so the solve sees the unscaled system;
  ///   * with an explicit single block (`set_field_partition({0})`) a scalar
  ///     1/||r|| really is formed and applied, but it multiplies both the
  ///     matrix and the right-hand side, so it leaves the least-squares
  ///     minimiser unchanged.
  /// Only a partition with two or more blocks changes the answer.
  bool scale_fields = true;
};

/// Spec §5.3. Stateful across the iterations of a window; `reset()` marks a
/// window boundary.

namespace Converged {

[[nodiscard]] inline ConvergenceMeasure Relative(double tol,
                                                 std::string field) {
  return detail::make_measure(MeasureKind::Relative, tol, std::move(field));
}
[[nodiscard]] inline ConvergenceMeasure Absolute(double tol,
                                                 std::string field) {
  return detail::make_measure(MeasureKind::Absolute, tol, std::move(field));
}
[[nodiscard]] inline ConvergenceMeasure AbsoluteOrRelative(double tol,
                                                           std::string field) {
  return detail::make_measure(MeasureKind::AbsoluteOrRelative, tol,
                              std::move(field));
}
[[nodiscard]] inline ConvergenceMeasure ResidualRelative(double tol,
                                                         std::string field) {
  return detail::make_measure(MeasureKind::ResidualRelative, tol,
                              std::move(field));
}

}  // namespace Converged

[[nodiscard]] inline ConvergenceCriterion operator&(ConvergenceMeasure a,
                                                    ConvergenceMeasure b) {
  ConvergenceCriterion c;
  c.add(std::move(a));
  c.add(std::move(b));
  return c;
}

[[nodiscard]] inline ConvergenceCriterion operator&(ConvergenceCriterion c,
                                                    ConvergenceMeasure m) {
  c.add(std::move(m));
  return c;
}


}  // namespace fuse
