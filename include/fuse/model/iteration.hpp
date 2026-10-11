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
#include <optional>
#include <string>
#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "fuse/base/error.hpp"
#include "fuse/base/rational.hpp"
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
  /// S7.2: the native Jacobian-free Newton-Krylov coupler
  /// (`fuse::num::solve()`, fuse/num/newton_krylov.hpp), selected the same
  /// way every quasi-Newton kind above is -- through
  /// `IterationRegionDecl::acceleration.kind` -- even though it replaces
  /// the ENTIRE sweep (Newton owns its own inner Krylov and line-search
  /// loop, where every kind above instead post-processes one Picard sweep
  /// per outer iteration). `docs/dev/newton-krylov.md`'s own design page
  /// (S7.1) anticipated exactly this: "the Newton solver itself (S7.2)
  /// adds its own AccelKind value and sets these flags when that value is
  /// chosen" -- `IterationRegionDecl::requires_residual_capability`/
  /// `requires_linearization_capability`, which `IterationBuilder::
  /// accelerate()`/`::newton()` below set automatically for exactly that
  /// reason, so a graph author cannot select Newton and forget the
  /// capability a Newton-solved region needs from every member.
  ///
  /// APPENDED AT THE END, not inserted in any more "natural" position:
  /// `fuse::model::yaml_io`'s `kAccelNames` table and
  /// `fuse::descriptor`'s binary encoding both index this enum
  /// positionally (REVIEW.md R-04's pattern), so a reordering would
  /// silently relabel every already-serialized graph's accelerator.
  Newton,
};

/// S7.2: which of SNESLineSearchBT/CP/Basic's own strategies the Newton
/// solver's line search uses. Mirrors `fuse::num::LineSearchKind`
/// (fuse/num/line_search.hpp) exactly -- declared again here, rather than
/// reused from there, for the SAME layering reason `AccelKind` itself is
/// declared in this header and not in fuse::num: the model layer must not
/// depend on the numerics layer (this file's own header comment). The
/// runtime wiring (a later step of this track) translates one into the
/// other at the one call site that constructs a `fuse::num::
/// NewtonKrylovOptions` from a `NewtonConfig`.
enum class LineSearchKind { BT, CP, Basic };

[[nodiscard]] constexpr const char* to_cstring(LineSearchKind k) noexcept {
  switch (k) {
    case LineSearchKind::BT:    return "BT";
    case LineSearchKind::CP:    return "CP";
    case LineSearchKind::Basic: return "Basic";
  }
  return "?";
}

/// S7.2: which Eisenstat-Walker forcing-term formula
/// (Eisenstat & Walker, SIAM J. Sci. Comput. 17, 1996) the Newton solver's
/// linear-solve tolerance follows. Mirrors `fuse::num::
/// EisenstatWalkerChoice` for the same reason `LineSearchKind` above does.
enum class EisenstatWalkerChoice { Constant, Choice1, Choice2 };

[[nodiscard]] constexpr const char* to_cstring(EisenstatWalkerChoice c) noexcept {
  switch (c) {
    case EisenstatWalkerChoice::Constant: return "Constant";
    case EisenstatWalkerChoice::Choice1:  return "Choice1";
    case EisenstatWalkerChoice::Choice2:  return "Choice2";
  }
  return "?";
}

/// S7.2: WHAT was asked of the Newton solver, named the same way
/// `AccelConfig` already does for the quasi-Newton ladder -- every numeric
/// default here is SNES's own default for the matching option
/// (`docs/dev/newton-krylov.md`'s own table names each one): SNES's own
/// forcing term clamps only at `eta_max`, never below (it can reach an
/// exact linear solve), so `eta_min`'s default is `0.0`, what makes that
/// default actually BE SNES's own -- the option itself still exists for a
/// caller who wants a nonzero floor, the possibility the task's own brief
/// named ("a cap eta_max = 0.9 and floor"), without presenting it as
/// something SNES does by default (CORRECTED: an earlier draft defaulted
/// this to `1e-4` as a judgement call presented alongside this struct's
/// genuinely-SNES defaults).
/// Read only when `IterationRegionDecl::acceleration.kind ==
/// AccelKind::Newton`; otherwise ignored, the same "declaration carries it,
/// nothing acts on it until selected" convention `window_scope` already
/// uses.
struct NewtonConfig {
  LineSearchKind line_search_kind = LineSearchKind::BT;
  EisenstatWalkerChoice forcing_choice = EisenstatWalkerChoice::Choice2;

  /// Eisenstat-Walker's own tuning constants (`fuse::num::
  /// EisenstatWalkerOptions`'s identically named fields carry the full
  /// citation and the over-solving-safeguard rationale).
  double forcing_gamma = 0.9;
  double forcing_alpha = 2.0;
  double eta_initial = 0.3;
  double eta_max = 0.9;
  double eta_min = 0.0;
  double forcing_safeguard_threshold = 0.1;
  double forcing_eta_constant = 0.1;

  double atol = 1e-50;
  double rtol = 1e-8;
  double stol = 1e-8;
  double divergence_tolerance = 1e4;
  int    max_nonlinear_iterations = 50;
  std::size_t max_function_evaluations = 10000;

  /// Line search's own tuning constants (`fuse::num::LineSearchOptions`'s
  /// identically named fields). Read only for the `line_search_kind` they
  /// apply to -- `armijo_alpha`/`lambda_min`/`max_backtracks` for BT,
  /// `secant_tol`/`max_secant_steps` for CP, `damping` for Basic.
  double armijo_alpha = 1e-4;
  double lambda_min = 1e-12;
  int    max_backtracks = 40;
  double secant_tol = 1e-10;
  int    max_secant_steps = 40;
  double damping = 1.0;

  /// GMRES(m)'s own restart and iteration budget.
  int gmres_restart = 30;
  std::size_t gmres_max_iterations = 1000;

  /// Rebuild the preconditioner every this-many Newton steps (and always
  /// before the first) -- SNES's own `-snes_lag_jacobian`.
  int preconditioner_lag = 1;

  /// Whether this region's Newton solve wants each member's own
  /// physics-based block-Jacobi preconditioner (`fuse::num::
  /// BlockJacobiPreconditioner`, composing every member's
  /// `LinearizationToken::apply_preconditioner`) rather than the
  /// identity. `true` here is what `IterationBuilder::newton()` reads to
  /// additionally set `requires_linearization_capability` -- a component
  /// cannot supply a preconditioner block without first being asked to
  /// linearise, so asking for one without the capability bit would be a
  /// silent fallback to the identity rather than the physics-based
  /// preconditioner the author actually requested.
  bool use_block_jacobi_preconditioner = false;
};

/// Time windows track: how a component's own sub-stepping relates to a
/// coupling window's end (`CouplingWindowScope`, fuse/schedule/window.hpp).
/// Chosen per component, never per region -- a mixed pair is the point.
enum class DtPolicy {
  /// The component keeps its own dt. A substep that would cross the window
  /// end is taken in full anyway (the straddling step): the part of it
  /// beyond the window end is advanced against the partner's trajectory
  /// EXTRAPOLATED past the window end (order >= 2, `WindowCoupler`'s
  /// `extrapolate_boundary_value`), and the whole substep is provisional --
  /// it is rolled back to its own start and recomputed with the partner's
  /// now-converged data as the FIRST act of the next window.
  Fixed,
  /// The component may shorten its own last substep in the window so that it
  /// lands exactly on the window end (preCICE's waveform-iteration
  /// convention). No straddling step and no extrapolation are ever needed
  /// for this component.
  Adjustable,
};

[[nodiscard]] constexpr const char* to_cstring(DtPolicy p) noexcept {
  switch (p) {
    case DtPolicy::Fixed:      return "Fixed";
    case DtPolicy::Adjustable: return "Adjustable";
  }
  return "?";
}

/// Time windows track: which classical waveform-relaxation scheme a
/// coupling window's own sweep loop runs (Gander & Vandewalle, "Analysis of
/// the parareal time-parallel time-integration method", SIAM J. Sci.
/// Comput. 29(2), 2007, and the earlier waveform-relaxation literature it
/// surveys -- these two names are exactly the classical Jacobi and
/// Gauss-Seidel iterations applied to the window's own coupled fixed-point
/// map, not FUSE-specific terms). Chosen per REGION (`CouplingWindowScope`
/// carries it, not per-channel): every member of one window sweeps under
/// the SAME scheme.
enum class WaveformIteration {
  /// Every field transfer inside the window's own group is loop-carried,
  /// whatever `Channel::lag` the user wrote: every member reads every
  /// partner's PREVIOUS iterate for the whole sweep, exactly as
  /// `WindowCoupler::sweep_window()` does today (its own `delivered`
  /// waveforms are rebuilt only BETWEEN sweeps, from the just-accelerated
  /// iterate, never mid-sweep). The default: it needs no `.lag()` at all to
  /// be schedulable, since neither direction's own field transfer is ever
  /// required to have run earlier in the SAME sweep.
  Jacobi,
  /// Exactly the field transfers the user declared `.lag(...)` on are
  /// loop-carried; every other direction is unlagged and must be delivered
  /// by an EARLIER instruction in the same sweep, same as an ordinary
  /// (non-window) implicit coupling group today. A two-way window with no
  /// lagged transfer at all is refused: with neither direction cut, the
  /// group closes an unbroken cycle with nothing to break it, exactly as
  /// an ordinary iteration region with no lag does today.
  GaussSeidel,
};

[[nodiscard]] constexpr const char* to_cstring(WaveformIteration w) noexcept {
  switch (w) {
    case WaveformIteration::Jacobi:      return "Jacobi";
    case WaveformIteration::GaussSeidel: return "GaussSeidel";
  }
  return "?";
}

/// One member's declared role inside a coupling window
/// (`CouplingWindowScope::policies`).
struct WindowMemberPolicy {
  std::string component;  ///< must name a `ComponentDecl` in the graph
  DtPolicy    policy = DtPolicy::Adjustable;
};

/// Time windows track (spec-in-progress §5.4): the opt-in payload that turns
/// a multi-clock `IterationRegionDecl` into a coupling window instead of the
/// contiguous single-clock loop `schedule()` otherwise requires
/// (`E-SCHED-REGION`; see `fuse::validate_coupling_window`,
/// `fuse/schedule/window.hpp`).
///
/// LIVES HERE, NOT IN `fuse/schedule/window.hpp`. It is a member of
/// `IterationRegionDecl` (`fuse/model/coupling_graph.hpp`), and
/// `coupling_graph.hpp` already includes this header for `DtPolicy` and
/// `ConvergenceCriterion` -- putting the type in `schedule/window.hpp`
/// instead, which itself includes `coupling_graph.hpp` for `CouplingGraph`,
/// would be a circular include. `fuse/schedule/window.hpp`'s free functions
/// (`window_length`, `window_region_members`, `validate_coupling_window`)
/// consume this type; they do not define it.
struct CouplingWindowScope {
  std::optional<std::string> window_component;
  std::optional<Rational>    explicit_interval;

  /// Every component that owns either end of one of the region's channels
  /// must appear here exactly once. `validate_coupling_window()` checks
  /// this; nothing here does, so an incomplete scope can be built
  /// incrementally with `.policies.push_back(...)` before validation.
  std::vector<WindowMemberPolicy> policies;

  /// Order of the temporal interpolant/extrapolant at the window boundary
  /// (`WindowCoupler`'s Waveform basis and its extrapolation degree). Must
  /// be >= 2 per the user's ruling ("interpolation and extrapolation are at
  /// least second order"); `Adjustable`/`Fixed` do not change this floor.
  int order = 2;

  /// Time windows track: which classical waveform-relaxation scheme this
  /// window's own sweep loop runs. Defaults to `WaveformIteration::Jacobi`
  /// -- the scheme `WindowCoupler` already implements today, and the one
  /// that needs no `.lag()` declaration at all to be schedulable.
  WaveformIteration iteration = WaveformIteration::Jacobi;

  [[nodiscard]] std::optional<DtPolicy> policy_for(
      std::string_view component) const {
    for (const WindowMemberPolicy& p : policies) {
      if (p.component == component) return p.policy;
    }
    return std::nullopt;
  }
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
