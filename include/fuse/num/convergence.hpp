#pragma once

#include <cmath>
#include <cstddef>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

/// Spec §5.3: the convergence measure carried by an `IterationRegion`.
///
///     g.iterate({fsi_channels})
///        .until(Converged::Relative(1e-6, "T_wall") &
///               Converged::Absolute(1e-8, "flux"))
///
/// These predicates decide when a coupled fixed-point iteration stops, so both
/// directions of error are quiet: too loose returns an unconverged answer that
/// looks converged, too tight never terminates. Every policy decision below is
/// therefore stated explicitly and pinned by a test in
/// test/unit/convergence_test.cpp.
///
/// **AND vs OR.** There are two combining rules and they are deliberately
/// different:
///   * *Within* one measure, `AbsoluteOrRelative` is an OR -- either branch
///     passing satisfies it.
///   * *Across* measures, `ConvergenceCriterion` is an AND -- every measure
///     must hold before the window advances. This matches preCICE's documented
///     behaviour for multiple convergence measures, and is what `&` in spec
///     §5.3 means. The two differ by orders of magnitude in practice, so the
///     operator is `&` and never `|`.
///
/// **Zero reference.** A relative test divides by a norm that can be exactly
/// zero -- a field that starts at zero is an ordinary case, not a pathological
/// one. Policy: a zero reference makes the relative branch fall back to the
/// absolute comparison against the same tolerance. It never yields NaN and
/// never accepts a large residual. The caveat is that the tolerance is then
/// read as a dimensional quantity rather than a ratio; a caller that cannot
/// accept that should supply a fixed non-zero reference of its own choosing.
///
/// **Non-finite norms.** A diverging iteration produces inf or NaN norms. That
/// is a runtime outcome, not a programmer error, so it is reported as NOT
/// converged rather than throwing. Note that this is not automatic: the
/// natural-looking `!(residual > tol)` is *true* for NaN and would report a
/// diverged run as converged. Every branch here is written as `x <= tol`,
/// which is false for NaN, and non-finite inputs are additionally rejected up
/// front so that e.g. residual/inf == 0 cannot pass either.
///
/// **Programmer errors** (a negative norm, a non-positive or non-finite
/// tolerance, a residual vector whose length does not match the number of
/// measures) throw. They are not `assert`s: `assert` compiles out under
/// -DNDEBUG, and a silently mis-sized residual vector is exactly the failure
/// that would report a converged run.
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
  std::ostringstream os;
  os.imbue(std::locale::classic());
  os << v;
  return os.str();
}

}  // namespace detail

struct ConvergenceMeasure {
  MeasureKind kind = MeasureKind::Relative;
  double      tolerance = 1e-6;
  std::string field;

  /// True when this measure alone considers the iteration converged.
  ///
  /// `reference_norm` is ignored by Absolute and used by every other kind.
  /// See the header comment for the zero-reference, non-finite and AND/OR
  /// policies. Throws std::invalid_argument on a non-positive or non-finite
  /// tolerance, or on a negative (finite) norm -- neither can arise from
  /// fuse::num::norm_l2 and friends, so both indicate a caller bug that must
  /// not be swallowed in a release build.
  [[nodiscard]] bool satisfied(double residual_norm,
                               double reference_norm) const {
    if (!(tolerance > 0.0) || !std::isfinite(tolerance)) {
      throw std::invalid_argument(
          "ConvergenceMeasure::satisfied: tolerance must be finite and "
          "positive, got " +
          detail::format_convergence_double(tolerance) + " for field '" +
          field + "'");
    }
    // Divergence, not a bug: never converged.
    if (!std::isfinite(residual_norm) || !std::isfinite(reference_norm)) {
      return false;
    }
    if (residual_norm < 0.0 || reference_norm < 0.0) {
      throw std::invalid_argument(
          "ConvergenceMeasure::satisfied: norms must be non-negative, got "
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
      throw std::invalid_argument(
          std::string{"ConvergenceCriterion::"} + fn + ": expected " +
          std::to_string(measures_.size()) +
          " residual and reference norms, got " +
          std::to_string(residual_norms.size()) + " and " +
          std::to_string(reference_norms.size()));
    }
  }

  std::vector<ConvergenceMeasure> measures_;
};

namespace detail {

/// Validating factory shared by the named constructors below. The same check
/// runs again in satisfied(), because ConvergenceMeasure is an aggregate and
/// can be brace-initialised past this function.
[[nodiscard]] inline ConvergenceMeasure make_measure(MeasureKind kind,
                                                     double tol,
                                                     std::string field) {
  if (!(tol > 0.0) || !std::isfinite(tol)) {
    throw std::invalid_argument(
        std::string{"Converged::"} + to_cstring(kind) +
        ": tolerance must be finite and positive, got " +
        format_convergence_double(tol) + " for field '" + field + "'");
  }
  return ConvergenceMeasure{kind, tol, std::move(field)};
}

}  // namespace detail

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
