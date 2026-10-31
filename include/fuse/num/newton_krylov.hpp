#pragma once

/// A native, Jacobian-free Newton-Krylov (JFNK) coupler on
/// fuse::linalg, replicating the mechanisms that make PETSc's SNESNEWTONLS
/// robust in practice -- without depending on PETSc (see
/// docs/dev/newton-krylov.md's design page).
///
/// EXACTLY SIX MECHANISMS, nothing else from SNES (no trust region, no
/// NGMRES, no nonlinear preconditioning):
///   1. MFFD Jacobian action (MffdJacobianAction) -- Walker-Pernice's
///      differencing parameter (Pernice & Walker, "NITSOL: A Newton
///      Iterative Solver for Nonlinear Systems", SIAM J. Sci. Comput. 19,
///      1998; PETSc's MATMFFD_WP).
///   2. The linear solve: fuse::linalg::gmres over a MatrixFreeOperator
///      wrapping (1), right-preconditioned by (6) or the identity.
///   3. Eisenstat-Walker forcing (eisenstat_walker_eta) -- Eisenstat &
///      Walker, "Choosing the Forcing Terms in an Inexact Newton Method",
///      SIAM J. Sci. Comput. 17, 1996.
///   4. Line search (fuse/num/line_search.hpp) -- BT/CP/Basic.
///   5. Named convergence and divergence tests (NewtonStatus) -- SNES's
///      own taxonomy (SNESConvergedReason), reproduced as an enum rather
///      than a bare bool so no outcome is ever silent.
///   6. A lagged, block-Jacobi preconditioner (BlockJacobiPreconditioner)
///      -- the physics-based preconditioning: each region
///      member's own LinearizationToken::apply_preconditioner
///      (fuse/runtime/component.hpp), composed over its own block of the
///      stacked interface vector. Rebuilt every `preconditioner_lag`
///      Newton steps via a caller-supplied refresh callback.
///
/// LAYERING. fuse::num depends on fuse::model and fuse::linalg, never on
/// fuse::runtime (src/CMakeLists.txt's own dependency graph) -- this
/// header therefore knows nothing of `fuse::Component` or
/// `fuse::LinearizationToken`. `PreconditionerBlock::apply` below is the
/// SAME duck-typed, span-based shape `LinearizationToken::
/// apply_preconditioner` already uses (`std::span<const double> r,
/// std::span<double> z`), so a caller in `fuse::runtime` (the graph-level
/// wiring) adapts a real token into
/// one with a one-line lambda; this header never needs to see the token
/// type itself.
///
/// RANK AGREEMENT. Every quantity this
/// file's convergence/divergence tests branch on -- `||F||`, `||dx||`,
/// the GMRES residual -- is read from `DistVector::norm2()`/`dot()`,
/// which already reduce across every participant inside `comm.hpp`'s
/// `Reducer` (dist_vector.hpp's own header comment). No branch here ever
/// reads an owned-block-local quantity directly, so two participants
/// computing the SAME reduced value make the SAME decision at the SAME
/// iteration, by construction -- test/num/newton_krylov_test.cpp's own
/// two-partition case proves this empirically with a thread-synchronised
/// in-process `Reducer` that deadlocks (rather than silently disagreeing)
/// the moment the two sides' call sequences diverge, which is exactly what
/// a decision made from an unreduced local quantity would cause.

#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "fuse/base/result.hpp"
#include "fuse/linalg/dist_vector.hpp"
#include "fuse/linalg/gmres.hpp"
#include "fuse/num/line_search.hpp"

namespace fuse::num {

using fuse::Result;
using fuse::linalg::DistVector;

/// `sqrt(eps_mach)`, MATMFFD_WP's own default differencing parameter
/// (PETSc's `-mat_mffd_err`, documented default). A plain function rather
/// than a `constexpr` so `MffdOptions::error_rel`'s default member
/// initialiser can call it without requiring `std::sqrt` to be usable in a
/// constant expression, which the C++20 standard library does not
/// guarantee.
[[nodiscard]] inline double default_mffd_error_rel() noexcept {
  return std::sqrt(std::numeric_limits<double>::epsilon());
}

// ===========================================================================
// 1. MFFD Jacobian action.
// ===========================================================================

/// `F(x)`: the coupled residual a Newton step zeroes
/// (`fuse::FixedPointResidual` adapts
/// `fuse::Runner::region_fixed_point_residual()`'s span-based evaluate
/// into exactly this `DistVector`-based shape). Non-const `x` because an
/// implementation may need `x.update_ghosts()` first, the same reason
/// `fuse::linalg::MatrixFreeOperator::ApplyFn` takes its input
/// non-const.
using ResidualFn = std::function<Result<void>(DistVector& x, DistVector& Fx)>;

struct MffdOptions {
  /// The relative differencing parameter, `sqrt(eps_mach)` by default --
  /// PETSc's MATMFFD_WP uses the same default (its own `-mat_mffd_err`,
  /// documented as `sqrt(eps_mach)` when left unset).
  double error_rel = default_mffd_error_rel();
  /// A pure zero-division safeguard on `||v||_2` (the direction's own
  /// norm) before it divides the Walker-Pernice step -- NOT PETSc's
  /// `-mat_mffd_umin`, which is a parameter of the DIFFERENT differencing
  /// routine `MATMFFD_DS` (Dennis-Schnabel: `h = error_rel * u'a / ||a||^2`
  /// when `|u'a| > umin ||a||_1`, else a `umin`-scaled fallback -- a
  /// direction-norm floor with a different formula entirely).
  /// `MATMFFD_WP`'s own routine (`wp.c`) has no `umin` parameter at all
  /// (its own tunable is `-mat_mffd_compute_normu`, whether to recompute
  /// `||x||` every call rather than cache it -- unrelated to this floor).
  /// Named and valued accordingly: `min_direction_norm` is the smallest
  /// NORMAL double times a margin (`std::numeric_limits<double>::min() *
  /// 1e8`), chosen only to keep the divisor away from zero and from the
  /// denormal range, never to approximate any PETSc option. The floor is
  /// reached routinely, not just in a corner case: `fuse::linalg::gmres()`
  /// evaluates its OWN first matrix-vector product at the initial guess
  /// `d = 0` before any Arnoldi step runs, so `apply()`'s very first call
  /// on every Newton step sees `v = 0` exactly -- without a floor, `h`
  /// would be `+inf` (a nonzero numerator over an exact-zero denominator),
  /// and `x + h*v` would be `x + inf*0`, IEEE 754's own NaN. The actual
  /// Jacobian action in that case is mathematically exact regardless of
  /// which positive `h` is used: `v = 0` means `x + h*v == x`, so
  /// `F(x+h*v) - F(x) == 0` identically and `apply()` returns the zero
  /// vector whatever finite `h` this floor produces.
  double min_direction_norm = std::numeric_limits<double>::min() * 1e8;
};

/// One `apply()` call's own step, recorded so a test can confirm the
/// Walker-Pernice rule was actually followed -- `h = error_rel *
/// sqrt(1 + ||x||_2) / max(||v||_2, min_direction_norm)`, Pernice & Walker
/// (1998)'s own differencing parameter for `MATMFFD_WP`.
struct MffdRecord {
  double h = 0.0;
  double x_norm = 0.0;
  double v_norm = 0.0;
};

/// Wraps a `ResidualFn` into a matrix-free action `J(x) v ~= (F(x + h v) -
/// F(x)) / h` about a fixed linearisation point `(x, Fx)`, set by
/// `set_point()` once per Newton step (so `F(x)` is evaluated once, not
/// once per Krylov iteration -- `apply()` costs exactly one new `F`
/// evaluation each call). NOT thread-safe and not reentrant: `apply()`
/// mutates its own scratch buffers and trace, the same single-threaded
/// contract every `fuse::Component` method already carries.
class MffdJacobianAction {
 public:
  explicit MffdJacobianAction(ResidualFn f, MffdOptions opt = {});

  /// Sets the linearisation point. `fx` is `F(x)`, already known by the
  /// Newton loop (it is the current nonlinear residual) -- `set_point`
  /// never calls `f` itself.
  void set_point(const DistVector& x, const DistVector& fx);

  /// `y <- J(x) v`, one new evaluation of `f`. Returns whatever `Failure`
  /// `f` itself returns, unchanged (`NewtonStatus::DivergedFunctionError`,
  /// the caller's own translation). `x`/`fx` must have been set by
  /// `set_point()` first; calling before that is a programmer error
  /// refused with `E-NEWTON-MFFD-NO-POINT`.
  Result<void> apply(DistVector& v, DistVector& jv);

  [[nodiscard]] const std::vector<MffdRecord>& trace() const noexcept { return trace_; }
  void clear_trace() { trace_.clear(); }

  /// Total `f` evaluations this object has issued via `apply()` --
  /// NewtonKrylovSolver's own function-evaluation budget
  /// (`NewtonKrylovOptions::max_function_evaluations`) is charged from
  /// this, not re-derived.
  [[nodiscard]] std::size_t evaluations() const noexcept { return evaluations_; }

 private:
  ResidualFn f_;
  MffdOptions opt_;
  std::optional<DistVector> x_, fx_;
  std::vector<MffdRecord> trace_;
  std::size_t evaluations_ = 0;
};

// ===========================================================================
// 2. (the linear solve itself is exactly fuse::linalg::gmres; no wrapper
//    needed beyond the MatrixFreeOperator newton_krylov.cpp builds around
//    MffdJacobianAction::apply.)
// ===========================================================================

// ===========================================================================
// 3. Eisenstat-Walker forcing.
// ===========================================================================

enum class EisenstatWalkerChoice {
  /// A fixed `eta` every step -- no adaptivity. Exists as the simplest
  /// alternative, and as the control a test compares the
  /// other two choices against.
  Constant,
  /// Eisenstat & Walker's choice 1: `eta_k = |  ||F_k|| - ||F_{k-1} +
  /// J_{k-1} d_{k-1}||  | / ||F_{k-1}||` -- the actual linear residual
  /// from the PREVIOUS inexact Newton step, which `fuse::linalg::
  /// SolverReport::residual_norm` already reports (it is `||b - A x||` at
  /// `b = -F_{k-1}`, `x = d_{k-1}`, i.e. exactly `||F_{k-1} + J_{k-1}
  /// d_{k-1}||`).
  Choice1,
  /// Eisenstat & Walker's choice 2 (SNES's own default):
  /// `eta_k = gamma (||F_k|| / ||F_{k-1}||)^alpha`.
  Choice2,
};

[[nodiscard]] const char* to_cstring(EisenstatWalkerChoice c) noexcept;

struct EisenstatWalkerOptions {
  EisenstatWalkerChoice choice = EisenstatWalkerChoice::Choice2;
  /// SNES's own defaults for choice 2 (`-snes_ksp_ew_version 2`'s
  /// `gamma`/`-snes_ksp_ew_rtol0` family).
  double gamma = 0.9;
  double alpha = 2.0;
  /// `eta_0`, the forcing term for the FIRST Newton step, before any
  /// history exists to compute choice 1 or 2 from. SNES's own default
  /// (`-snes_ksp_ew_rtol0`).
  double eta_initial = 0.3;
  /// Hard cap -- SNES's own `-snes_ksp_ew_rtolmax`.
  double eta_max = 0.9;
  /// Floor. SNES's own forcing-term code clamps only at `eta_max`, not
  /// below -- its forcing term can reach exactly zero (an exact linear
  /// solve). The OPTION exists so a cap `eta_max = 0.9` and a floor can be
  /// chosen, but the DEFAULT is `0.0`
  /// precisely so that it is SNES's own default (CORRECTED: an earlier
  /// draft defaulted this to `1e-4` as a judgement call, presented
  /// alongside options that are genuinely SNES's own defaults, which
  /// understated how many of this file's defaults are actually
  /// unmodified SNES behaviour).
  double eta_min = 0.0;
  /// The over-solving safeguard (Eisenstat & Walker 1996, eq. 3.5-ish;
  /// SNES's own implementation): when `gamma * eta_{k-1}^alpha > 0.1`,
  /// `eta_k` is floored at `gamma * eta_{k-1}^alpha` rather than left at
  /// whatever the plain formula above gives -- without it, a Newton step
  /// that reduces `||F||` a lot can starve the NEXT linear solve of
  /// accuracy it still needs.
  double safeguard_threshold = 0.1;
  /// Choice::Constant's own fixed value.
  double eta_constant = 0.1;
};

/// `k` is the 0-based Newton iteration about to be taken (`k == 0` means
/// "choosing eta for the very first linear solve", which uses
/// `eta_initial` for every choice -- there is no history to compute
/// choice 1 or 2 from yet). `norm_fk`/`norm_fkm1` are `||F_k||`/
/// `||F_{k-1}||` (meaningless at `k == 0`, ignored there). `linear_residual_prev`
/// is choice 1's own `||F_{k-1} + J_{k-1} d_{k-1}||`; pass a negative value
/// when it is unavailable (choice 2 and Constant never read it).
[[nodiscard]] double eisenstat_walker_eta(const EisenstatWalkerOptions& opt, int k,
                                          double eta_prev, double norm_fk, double norm_fkm1,
                                          double linear_residual_prev);

// ===========================================================================
// 5. Named convergence and divergence tests.
// ===========================================================================

enum class NewtonStatus {
  Running,
  /// ||F|| <= atol.
  ConvergedAbsolute,
  /// ||F|| <= rtol * ||F_0||.
  ConvergedRelative,
  /// ||dx|| <= stol * ||x||.
  ConvergedStep,
  /// The function-evaluation budget (MFFD actions plus line-search
  /// trials) was exhausted.
  DivergedFunctionCount,
  /// The line search could not find an acceptable step.
  DivergedLineSearch,
  /// ||F|| grew past divergence_tolerance * ||F_0||.
  DivergedResidualGrowth,
  /// GMRES itself reported E-LINALG-NO-CONVERGENCE.
  DivergedLinearSolve,
  /// The Newton iteration budget was exhausted with no convergence test
  /// satisfied.
  DivergedMaxIterations,
  /// `F` itself returned a Failure.
  DivergedFunctionError,
};

[[nodiscard]] const char* to_cstring(NewtonStatus s) noexcept;
/// Running/Converged* -> false; Diverged* -> true. A caller deciding
/// whether to trust `x` on return reads this, not a string comparison.
[[nodiscard]] bool is_diverged(NewtonStatus s) noexcept;
[[nodiscard]] bool is_converged(NewtonStatus s) noexcept;

// ===========================================================================
// 6. Lagged, block-Jacobi preconditioner.
// ===========================================================================

/// One region member's own diagonal-block solve, `z <- M^-1 r`, over ITS
/// OWN slice `[offset, offset + length)` of the stacked Newton iterate --
/// the exact span-based shape `fuse::LinearizationToken::
/// apply_preconditioner` (fuse/runtime/component.hpp) already uses, so a
/// caller above fuse::runtime adapts a real token with a one-line lambda.
/// The identity (copy `r` to `z`) is always a legal `apply`, the same
/// "a component with no better choice may implement this as the
/// identity" licence `LinearizationToken::apply_preconditioner`'s own
/// comment grants.
struct PreconditionerBlock {
  std::size_t offset = 0;
  std::size_t length = 0;
  std::function<Result<void>(std::span<const double> r, std::span<double> z)> apply;
};

/// `M^-1 = blockdiag(M_1^-1, ..., M_m^-1)`, each `M_i^-1` exactly one
/// `PreconditionerBlock::apply`. Built once per preconditioner refresh
/// (NewtonKrylovOptions::preconditioner_lag decides how often that is);
/// `apply()` itself issues no new evaluation of anything -- it is a pure
/// local recombination of each block's own `apply`, over the OWNED
/// portion of the vector (matching
/// `fuse::Runner::region_fixed_point_residual()`'s single-participant
/// restriction; a cross-rank block-Jacobi preconditioner would need the
/// fixed-point residual itself lifted to a window-scoped,
/// multi-participant region).
class BlockJacobiPreconditioner {
 public:
  /// Refuses `E-NEWTON-PRECOND-COVERAGE` when the blocks do not exactly
  /// tile `[0, n)` with no gap and no overlap -- a Newton region whose
  /// block-Jacobi preconditioner silently leaves part of the vector
  /// un-preconditioned (identity there, with nobody having asked for
  /// that) is a worse failure than refusing to build it at all.
  [[nodiscard]] static Result<BlockJacobiPreconditioner> from_blocks(
      std::vector<PreconditionerBlock> blocks, std::size_t n);

  /// `y <- M^-1 x`, over `x.owned_size() == n`. Satisfies
  /// `fuse::linalg::Preconditioner`.
  [[nodiscard]] Result<void> apply(DistVector& x, DistVector& y) const;

  [[nodiscard]] std::size_t n() const noexcept { return n_; }
  [[nodiscard]] const std::vector<PreconditionerBlock>& blocks() const noexcept { return blocks_; }

 private:
  BlockJacobiPreconditioner(std::vector<PreconditionerBlock> blocks, std::size_t n)
      : blocks_(std::move(blocks)), n_(n) {}

  std::vector<PreconditionerBlock> blocks_;
  std::size_t n_ = 0;
};

/// Rebuilds `*target` from a caller-supplied factory at the current point
/// `(x, fx)`. Not `BlockJacobiPreconditioner`-specific -- the Newton
/// driver calls this at the lag boundary regardless of what it rebuilds,
/// including "nothing" (a `nullptr` RefreshFn is a legal no-refresh, the
/// identity preconditioner's own case).
using RefreshPreconditionerFn =
    std::function<Result<void>(const DistVector& x, const DistVector& fx)>;

/// `z <- M^-1 r`, type-erased so `NewtonKrylovSolver::solve()` need not be
/// templated on the preconditioner's concrete type. A `nullptr` means the
/// identity.
using PreconditionerApplyFn = std::function<Result<void>(DistVector& r, DistVector& z)>;

// ===========================================================================
// The driver.
// ===========================================================================

struct NewtonKrylovOptions {
  /// SNES's own defaults (`-snes_atol`, `-snes_rtol`, `-snes_stol`).
  double atol = 1e-50;
  double rtol = 1e-8;
  double stol = 1e-8;
  /// `-snes_divergence_tolerance`.
  double divergence_tolerance = 1e4;
  /// `-snes_max_it`.
  int max_nonlinear_iterations = 50;
  /// `-snes_max_funcs`.
  std::size_t max_function_evaluations = 10000;

  MffdOptions mffd;
  EisenstatWalkerOptions forcing;
  /// `rel_tol` is overwritten every Newton step with that step's own
  /// `eta_k` -- whatever is set here on entry is ignored; `restart` and
  /// `max_iterations` are read as given.
  fuse::linalg::GmresOptions gmres;
  LineSearchOptions line_search;
  /// Rebuild the preconditioner every `preconditioner_lag` Newton steps
  /// (and always before the first). 1 means "every step" (SNES's own
  /// `-snes_lag_jacobian 1` default); `-snes_lag_jacobian -2` style
  /// "never after the first" is `preconditioner_lag` set larger than
  /// `max_nonlinear_iterations`.
  int preconditioner_lag = 1;
};

/// One Newton step's own record -- NewtonKrylovSolver's own trace, read by
/// the tests and by a caller measuring the quadratic rate.
struct NewtonIterationRecord {
  int iteration = 0;
  double norm_f = 0.0;
  double eta = 0.0;
  std::size_t gmres_iterations = 0;
  double gmres_residual_norm = 0.0;
  std::vector<MffdRecord> mffd_steps;
  LineSearchStatus line_search_status = LineSearchStatus::FullStep;
  double line_search_lambda = 1.0;
  std::vector<double> line_search_trials;
  bool preconditioner_refreshed = false;
};

struct NewtonResult {
  NewtonStatus status = NewtonStatus::Running;
  int iterations = 0;
  std::size_t function_evaluations = 0;
  double initial_residual_norm = 0.0;
  double final_residual_norm = 0.0;
  std::vector<NewtonIterationRecord> trace;
};

/// The inexact Newton-Krylov loop itself. `x` is the initial guess on
/// entry and, whatever `NewtonResult::status` comes back, the last
/// iterate the loop produced on exit (NEVER left at a mid-step partial
/// value -- every exit path, converged or diverged, exits with `x` at a
/// point `F` was actually evaluated at).
///
/// `precond_apply`/`refresh_precond` may both be `nullptr`: no
/// preconditioning, no refresh, the plain unpreconditioned matrix-free
/// Newton-Krylov iteration.
[[nodiscard]] Result<NewtonResult> solve(const ResidualFn& f, DistVector& x,
                                         const PreconditionerApplyFn& precond_apply,
                                         const RefreshPreconditionerFn& refresh_precond,
                                         const NewtonKrylovOptions& opt = {});

// ===========================================================================
// The Newton substrate registry.
// ===========================================================================

/// Mirrors `fuse::NewtonSubstrate` (fuse/model/iteration.hpp) -- declared
/// again here, not reused from there, for the SAME layering reason this
/// header's own `LineSearchKind`/`EisenstatWalkerChoice` counterparts are
/// mirrored in the opposite direction by that header: `fuse::model`
/// sits BELOW `fuse::num` (tools/check-layering.py's own `LAYERS` list),
/// so `fuse::num` may depend on `fuse::model` directly -- but a
/// registry keyed on a model-layer type would make this header's own
/// public interface name a model-layer type for no reason internal to
/// what the registry does (dispatch on two possible solve functions),
/// the same "this header's own public interface should not need to know
/// about a layer it does not otherwise depend on" reasoning that keeps
/// every other enum here duplicated instead of `#include`d. The runtime
/// wiring (`src/runtime/iteration.cpp`, which DOES depend on both layers)
/// translates one into the other at its one call site, exactly the
/// pattern `to_num_line_search_kind()`/`to_num_ew_choice()` already set.
enum class NewtonSubstrate { Native, Petsc };

[[nodiscard]] constexpr const char* to_cstring(NewtonSubstrate s) noexcept {
  switch (s) {
    case NewtonSubstrate::Native: return "Native";
    case NewtonSubstrate::Petsc:  return "Petsc";
  }
  return "?";
}

/// One substrate's own solve entry point, type-erased to the identical
/// signature `fuse::num::solve()` above already has -- a caller that
/// dispatches through the registry calls either substrate the same way,
/// with no `if` on which one it got.
///
/// `comm_f`: an `MPI_Comm_c2f` handle for the
/// region's own participant communicator, `0` when there is none (a
/// single-rank region, or no communicator configured at all) -- the ONE
/// thing `fuse::num`'s own comm-free abstraction (`fuse::linalg::
/// Reducer`/`DistVector`, which `x` already carries a working reduction
/// through) cannot hand a substrate that needs a REAL `MPI_Comm` for its
/// own objects, which is exactly `fuse::petsc::solve()`'s situation --
/// `SNES`/`Vec`/`Mat` are PETSc's own types, built directly on a
/// communicator, never on this library's `Reducer` facade. The Native
/// substrate (`fuse::num::solve()` itself, registered below) ignores
/// this parameter entirely: every reduction its own GMRES/line-search/
/// forcing-term machinery needs already goes through `x`'s own `Reducer`
/// (`dist_vector.hpp`'s own `dot()`/`norm2()`/`norm_inf()`), so adding
/// this parameter changed NO behaviour on that substrate -- only its own
/// call signature, to keep one registry entry type both substrates
/// implement.
using NewtonSolveFn = std::function<Result<NewtonResult>(
    const ResidualFn&, DistVector&, const PreconditionerApplyFn&,
    const RefreshPreconditionerFn&, const NewtonKrylovOptions&, int comm_f)>;

/// Registers (or replaces) the solve entry point for `kind`. Called by
/// `fuse::petsc::install()` (fuse/petsc/solve.hpp) to register
/// `NewtonSubstrate::Petsc`; `NewtonSubstrate::Native` is registered by
/// this library ITSELF (see `newton_substrate()`'s own comment) and
/// needs no call here unless a caller deliberately wants to replace it
/// (e.g. a test double).
void register_newton_substrate(NewtonSubstrate kind, NewtonSolveFn fn);

/// The registered solve entry point for `kind`, or `nullptr` if none is
/// registered. `NewtonSubstrate::Native` is ALWAYS non-null: it is
/// registered deterministically on first use, by a function-local
/// static inside this function's own definition (src/num/
/// newton_krylov.cpp) -- never by a static initialiser in a .cpp that
/// might not be linked into a given binary at all (a build that never
/// references anything else in that translation unit would then never
/// run the initialiser, and `fuse::num::solve()` itself, the function
/// every other caller in this file already depends on directly, is
/// certainly linked wherever this header is used). `NewtonSubstrate::
/// Petsc` is non-null only after `fuse::petsc::install()` has run
/// (requires `FUSE_ENABLE_PETSC`); a caller that gets `nullptr` back for
/// it is what lets `Runner::initialize()` (src/runtime/iteration.cpp)
/// refuse `E-NEWTON-PETSC-UNAVAILABLE` naming the region, rather than
/// dereferencing a missing function.
[[nodiscard]] const NewtonSolveFn* newton_substrate(NewtonSubstrate kind);

}  // namespace fuse::num
