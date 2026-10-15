#pragma once

// AgreementBounds track: the native-vs-PETSc agreement bound for the
// Brusselator's own window-Newton coupling (docs/dev/newton-krylov.md
// Section 9), replacing a chosen constant (first a 20x margin on a mistaken
// proxy, then a measured-but-still-chosen inflation of a LOWER bound on
// ||J^-1|| -- rejected: a multiple of a lower bound proves nothing about
// an upper bound, no matter how it is flagged) with an A-POSTERIORI
// CALIBRATION measured on the SAME configuration the check runs, never a
// different one.
//
// WHY NOT A DERIVED ||J^-1||, like dn_heat's. dn_heat's bound (examples/
// dn_heat/agreement_bound.hpp) is exact because its own residual is
// affine with a known, closed-form per-station Jacobian. The
// Brusselator's window-Newton residual is nonlinear (the reaction term)
// and spans multiple window substeps (docs/dev/newton-krylov.md Section 9.1);
// no closed-form Jacobian-inverse bound was derived for it, and the
// Runner's public API exposes only ||F|| at each Newton iterate
// (IterationStats::residual_history), never the iterate itself or a
// Jacobian action -- there is nothing here to derive an exact bound
// from.
//
// THE CALIBRATION, instead. `test/petsc/brusselator_newton_test.cpp`'s
// own `measure_native_reference_error()` solves the SAME window three
// times, natively (no PETSc involved in the calibration itself):
//   - at the CHECK's own `NewtonConfig::rtol` (the model default, 1e-8,
//     unoverridden -- the SAME tolerance `ConvergesAgreeingWithNative`'s
//     own native/PETSc solves use), giving the state the real check
//     compares against PETSc;
//   - at two successively tighter `rtol` values (1e-13, 1e-14), to
//     obtain a REFERENCE solution and confirm it has stopped moving
//     (the finite-difference Jacobian's own noise floor: tightening
//     `rtol` further stops changing the converged state once the
//     Newton iteration's own achievable precision is reached).
// `e_n`, the max-norm difference (per field) between the check-tolerance
// state and the reference, is the MEASURED error of an rtol-converged
// native solve of THIS problem -- not assumed, not bounded analytically,
// read directly off two real solves.
//
// THE SCALING ASSUMPTION -- stated, not hidden. `e_n` is the native
// solver's own error at its OWN final residual, ||F_final,native||. The
// PETSc solve's error is taken to scale the SAME way: the error of a
// Newton iterate x satisfies x - x* = J(x*)^-1 F(x) to first order (the
// SAME identity dn_heat's own bound uses exactly, here only approximately
// since F is nonlinear) -- both solvers' iterates are near the SAME root
// x*, so the SAME operator J(x*)^-1 relates each one's own error to its
// own final residual, giving error_petsc / error_native ~= ||F_final,
// petsc|| / ||F_final,native||, i.e. error_petsc ~= e_n * (||F_final,
// petsc|| / ||F_final,native||). The bound on the two solvers' own
// DISAGREEMENT is then the sum of both errors (triangle inequality):
//
//   bound = e_n * (1 + ||F_final,petsc|| / ||F_final,native||)
//           + kMeasuredInterSolverRelativeDifference * ||native solution||
//
// THIS ASSUMPTION CAN BE FALSE. If the PETSc solve's own Newton
// trajectory approaches the root from a direction where J(x*)^-1 acts
// with a LARGER norm than it does along the native trajectory this
// calibration measured, the real PETSc error could exceed what this
// formula predicts, and the check could PASS when the two solutions
// actually disagree by more than a sound bound would allow. A check
// failure under this bound means either a real disagreement or that this
// scaling assumption itself has broken down for the given run; either
// way it is signal, not noise, and the bound must not be loosened to
// silence it.
//
// ROUND 3 (the owner's own correction to round 2): the SEPARATE
// `kMeasuredInterSolverRelativeDifference * ||native solution||` term
// below is REMOVED from the asserted bound. It is docs/dev/newton-
// krylov.md Section 10.3's measured FIRST-NEWTON-STEP residual difference
// between the native and PETSc GMRES implementations on SYNTHETIC
// quadratic/atan unit problems -- a measurement about how the two
// Krylov solvers' own internal arithmetic differs ONE STEP IN, not a
// bound on the difference between two CONVERGED solutions of the
// Brusselator's own problem. The calibrated error term above already
// accounts for every way `x_n` and `x_p` can differ, through the
// scaling assumption's own identity (`x - x* = J(x*)^-1 F(x)`, applied
// to each solver's own terminal residual) -- adding a second, unrelated
// term on top double-counts a quantity this problem's own calibration
// already measures directly.
//
// ROUND-OFF FLOOR -- examined, NOT defensibly derived, unlike dn_heat's
// (examples/dn_heat/agreement_bound.hpp's own "ROUND-OFF FLOOR"
// section). dn_heat's committed state update is a two-operation
// floating-point chain per station (Higham's standard model applies
// directly). The Brusselator's own committed state update
// (`evaluate_window_region_residual(..., commit=true)`) re-integrates
// the WHOLE window at the converged iterate: multiple backward-Euler
// diffusion SOLVES (a sparse linear system each substep, `species_
// core.hpp`'s own `DiffusionSolver`), a nonlinear reaction term
// evaluated at several points within the window, and conservative remap
// between the U and V grids. A rigorous backward-error bound for a
// sparse linear solve takes the form `gamma_{c*nnz} * ||A|| * ||x||`
// (Higham, "Accuracy and Stability of Numerical Algorithms", Ch. 10),
// `c` a fill-in-dependent constant this track did not derive from
// `DiffusionSolver`'s own factorisation, and the window compounds up to
// 11 such solves (U's own `window_sample_capacity`, docs/dev/newton-
// krylov.md Section 9.1) with remap interpolation between them. Deriving `||A||`
// and the fill-in constant from the code rather than guessing would be
// needed for a defensible number, and this track did not do that work --
// so NO round-off floor is added here. If the host's own observed
// disagreement sits consistently above the calibrated bound by a SMALL,
// stable margin (rather than scaling with the residual ratio the way the
// error term does), that margin is a candidate round-off floor to derive
// properly, not to tune to.

#include <cmath>

namespace brusselator_bound {

/// The SHARP calibrated agreement bound (this header's own derivation
/// above): `e_n * (1 + f_final_petsc / f_final_native)`, with NO
/// separate inter-Krylov term (see this header's own "ROUND 3" section
/// for why it was removed).
///
/// `e_n`: the native solver's own measured error at the check's `rtol`,
/// against a tight reference solve of the IDENTICAL configuration
/// (`measure_native_reference_error()`, test/petsc/
/// brusselator_newton_test.cpp).
/// `f_final_native`/`f_final_petsc`: each solver's own converged ||F||
/// (`RunResult::residual_history.back()`). Precondition: `f_final_native
/// > 0` (a solver that reports exactly zero residual has no well-defined
/// residual ratio; this problem's own finite-difference Newton iteration
/// has never been observed to report exactly zero).
[[nodiscard]] inline double agreement_bound(double e_n, double f_final_native,
                                            double f_final_petsc) {
  const double residual_ratio = f_final_petsc / f_final_native;
  return e_n * (1.0 + residual_ratio);
}

// ---------------------------------------------------------------------
// LEGACY (round 2's own formula) -- kept ONLY so the real check can
// print it, labelled `[agreement-bound legacy]`, alongside the sharp
// bound above, informationally. NEVER asserted against -- this header's
// own "ROUND 3" section is why.
// ---------------------------------------------------------------------

/// docs/dev/newton-krylov.md Section 10.3's own measured relative difference
/// between the native and PETSc GMRES/Arnoldi implementations, on a
/// synthetic quadratic/atan problem -- round 2's own additional term,
/// kept here only for the legacy formula below.
inline constexpr double kMeasuredInterSolverRelativeDifference = 3.9e-8;

/// Round 2's own formula, verbatim: the SAME calibrated error term,
/// plus the inter-Krylov term this round's own derivation found to be
/// measuring the wrong kind of quantity for this purpose. See this
/// header's own top-of-file "ROUND 3" section.
[[nodiscard]] inline double legacy_agreement_bound(double e_n, double f_final_native,
                                                    double f_final_petsc,
                                                    double native_solution_norm) {
  const double residual_ratio = f_final_petsc / f_final_native;
  const double error_term = e_n * (1.0 + residual_ratio);
  const double gmres_term =
      kMeasuredInterSolverRelativeDifference * native_solution_norm;
  return error_term + gmres_term;
}

}  // namespace brusselator_bound
