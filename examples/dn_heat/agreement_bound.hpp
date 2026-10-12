#pragma once

// AgreementBounds track, round 4 (the owner's own correction to round 3):
// the SHARP native-vs-PETSc agreement bound for dn_heat's affine graph,
// factored out of examples/dn_heat/main.cpp's `--newton-petsc` check so a
// tier-0 unit test (test/verification/dn_heat_agreement_bound_test.cpp)
// can hand-build numbers and prove the derivation, rather than asserting
// it only end to end through a PETSc run this sandbox cannot execute
// (KNOWN_ISSUES.md D32).
//
// WHAT WAS WRONG WITH ROUND 2's BOUND, by the owner's own diagnosis.
// `||J^-1|| (rtol_n ||F_0||_n + rtol_p ||F_0||_p)` bounds the disagreement
// using the SOLVERS' OWN STOPPING CRITERION (`rtol * ||F_0||`), not what
// either solver's own TERMINAL residual actually measures -- and Newton's
// own quadratic convergence routinely overshoots that criterion by many
// orders of magnitude (measured: a terminal ||F|| of 4.159e-11 against a
// stopping level near `rtol * ||F_0|| ~ 1e-8 * 56.8 ~ 5.7e-7`). Scaling by
// the stopping criterion instead of the actual terminal residual made the
// bound many orders of magnitude looser than the quantity it was meant to
// certify. The separate `3.9e-8 * ||native solution||` term was ALSO the
// wrong kind of quantity for this purpose: it is docs/dev/newton-krylov.md
// §10.3's measured first-Newton-step residual difference between the
// native and PETSc GMRES implementations on SYNTHETIC quadratic/atan unit
// problems -- a measurement about how the two Krylov solvers' own internal
// arithmetic differs ONE STEP IN, not a bound on the difference between
// two CONVERGED Newton solutions of dn_heat's own problem.
//
// WHAT WAS WRONG WITH ROUND 3's BOUND, by the owner's own reading of
// `main.cpp`. Round 3's identity (below, still exact for ONE period)
// used only `Outcome::last_residuals.back()`, the LAST of `kPeriods`
// (=20) periods' own terminal residual -- but `x - x* = J^-1 F(x)` being
// exact for affine `F` requires comparing the SAME map `F`, which only
// holds cleanly when both solvers start period `k` from an IDENTICAL
// `x`. That is true of period 1 (both substrates start from the
// IDENTICAL initial state), but from period 2 on, each period's own
// Newton solve starts from whatever the PREVIOUS period committed --
// and native and PETSc committed slightly different values, by exactly
// the previous period's own disagreement. A bound built from period 20
// alone said nothing about whatever periods 1..19 may have contributed
// on top of that. The "PER-PERIOD RECURSION" section below derives the
// corrected bound; it turns out, once derived from the actual commit
// formulas, that it NEEDS every period's own terminal residual, not an
// assumption that the carried-forward disagreement cannot grow -- see
// that section for why the station's own contraction factor `r_i` does
// NOT, by itself, bound how the carried disagreement amplifies.
//
// THE SHARP BOUND (exact for ONE period, given identical inputs).
// `fixed_point_residual_test.cpp`'s own header comment derives
// dn_heat's fixed-point residual exactly: `F(x) = x - G(x)` is affine
// in the stacked interface vector `x = (q, u)`, with a per-station 2x2
// Jacobian `J = {{1, r_i}, {-1, 1}}`, `r_i` the station's own
// conductivity ratio (`dn_component.hpp`'s `kStripRatio`, in [0.50, 0.96]).
// For an affine `F(x) = Jx + c` with a UNIQUE root `x*` (`Jx* + c = 0`):
// `x - x* = J^-1 (F(x) - F(x*)) = J^-1 F(x)` EXACTLY, for ANY `x` -- so two
// approximate roots `x_n`, `x_p`, both solving the SAME period's `F`,
// satisfy, by the triangle inequality,
//     ||x_n - x_p|| <= ||J^-1|| (||F(x_n)|| + ||F(x_p)||)
// with `F(x_n)`/`F(x_p)` each solver's own TERMINAL residual (what its
// Newton iteration actually achieved), never the stopping criterion that
// merely bounds them from above. No separate inter-Krylov term is added:
// the identity above already accounts for EVERY way `x_n` and `x_p` can
// differ, through their own residuals, regardless of which Krylov/
// line-search internals produced them -- and it is EXACT, not merely to
// first order, because `F` really is affine here. This is the per-period
// building block the recursion below sums over all `kPeriods` periods.
//
// ||J^-1||_inf: `J^-1 = 1/(1+r_i) * {{1, -r_i}, {1, 1}}` (adjugate of `J`
// divided by `det(J) = 1 + r_i`). Row sums of absolute values: row 0 is
// `(1+r_i)/(1+r_i) = 1`; row 1 is `2/(1+r_i)`. The infinity norm is the
// larger of the two, and `2/(1+r_i)` is DECREASING in `r_i`, so it is
// maximised at the smallest station ratio, `r_i = 0.50`: `||J^-1||_inf =
// 2/1.5 = 4/3` EXACTLY -- no rounding, since 4/3 is already the
// exhaustive maximum over the four strips `dn_component.hpp` defines
// (`kStripRatio = {0.50, 0.75, 0.90, 0.96}`, all `>= 0.50`).
//
// NORM MISMATCH, resolved explicitly. Each solver's own terminal residual
// is reported in the EUCLIDEAN (2-) norm (`DistVector::norm2()`, src/num/
// newton_krylov.cpp); the bound above needs ||F|| in the INFINITY norm
// the per-station `disagreement()` check uses. Since ||v||_inf <= ||v||_2
// for any finite vector, using each solver's own 2-norm terminal residual
// in place of its (unmeasured) infinity-norm residual is a valid, if
// slightly conservative (never underestimating), substitution, stated
// here so the norm choice is not silently assumed.
//
// WHICH RESIDUAL. `Outcome::last_residuals` is cleared and refilled
// every period (`run_region_newton()`, src/runtime/iteration.cpp:
// `st.residual_history.clear();` at the top of the function) -- by the
// time a `kPeriods`-period run finishes, only the LAST period's own
// trace survives there. This is neither a true maximum over all periods
// nor a per-period series; it is the terminal residual of the SPECIFIC
// period whose own committed Newton solve produced `u_final`
// (`run_region_newton()`'s own verification sweep lands exactly there).
// It remains the right quantity for `legacy_agreement_bound()` below
// (unchanged from round 2/3) and for round 3's own single-period
// identity above, but the `--newton-petsc` check compares `u_final`
// AFTER the full `kPeriods`-period run, so the ASSERTED bound needs
// every period's own terminal residual, not the last one alone --
// `Outcome::per_period_final_residual` (`main.cpp`, populated by
// calling `Runner::run(components, 1)` once per period instead of once
// for all of them, since `iteration_stats()` only ever describes "the
// most recent period").
//
// PER-PERIOD RECURSION (round 4). Let `e_k` be the max-norm disagreement
// between the two substrates' own committed states after period `k`,
// and let `A` be the map carrying period `k-1`'s own committed state
// into period `k`'s own Newton STARTING point (its initial iterate).
// `A` is read directly off `dn_component.hpp`'s own update formulas:
//     DirichletSlab::advance: q_[i] = k1_eff(i, u_in_[i]) * (T1 - u_in_[i]) / kL1;
//     NeumannSlab::advance:   u_[i] = T2 + q_in_[i] * kL2 / kK2;
// and the fact that nothing else writes `u_in_`/`q_in_`/`q_`/`u_` between
// periods: whatever period `k-1` last committed into these registered
// arrays is bit-for-bit what period `k`'s own Newton iteration starts
// from. The carry-over `A` is therefore the IDENTITY, `||A||_inf = 1`
// exactly -- which already satisfies fix (A)'s own `||A||_inf <= 1`
// requirement, so the recursion
//     e_k <= ||A||_inf e_{k-1} + ||J^-1||_inf (F_n,k + F_p,k)
// telescopes (`e_0 = 0`: both substrates start from the IDENTICAL
// initial state) to
//     e_kPeriods <= ||J^-1||_inf * sum_{k=1}^{kPeriods} (F_n,k + F_p,k),
// the sum over EVERY period's own terminal residual. This is the bound
// asserted below.
//
// A separate observation, NOT relied on for the assertion above: both
// `advance()` formulas are plain ASSIGNMENTS depending only on the
// CURRENT period's own `t` (through `T1`/`T2`) and the CURRENT Newton
// iterate's own incoming value -- never on a value carried from a
// previous period except as that iteration's STARTING GUESS. Since the
// sharp per-period identity `x - x*_k = J^-1 F(x)` holds for ANY `x`
// (not only a well-converged one), period `k`'s own committed `x_k`
// satisfies it regardless of which `x_0` the solver started from, which
// suggests the TRUE per-period amplification may be zero rather than
// merely bounded by 1 (i.e. that only the LAST period's own terms would
// in fact suffice). That stronger claim is plausible but not what this
// header asserts -- the sum above, from the literal `||A||_inf = 1`
// carry-over, is strictly no tighter an assumption than this header can
// defend from the code alone, and is never looser than the stronger
// claim would be.
//
// ROUND-OFF FLOOR -- examined, not added. dn_heat's own committed
// `u_final` is produced by a short floating-point chain
// (`dn_component.hpp`'s `NeumannSlab::advance()`: `u_[i] = T2 +
// q_in_[i] * kL2 / kK2`, and `kL2 == kK2 == 1.0` EXACTLY, so IEEE 754
// makes `x * 1.0`/`x / 1.0` exact -- the only operation that can round at
// all is the single addition `T2 + q_in_[i]`). Higham's standard model
// gives `|fl(T2 + q_in) - (T2 + q_in)| <= u * (|T2| + |q_in|)`, `u` the
// unit roundoff (2^-53 for `double`). `T_outer_2(t) = 280.0 + 15.0 *
// sin(...)` bounds `|T2| <= 295`; `q_in_[i] = k1_eff(i, u_in_[i]) *
// (T1 - u_in_[i])` with `k1_eff <= 0.96` (`kStripRatio`'s own maximum)
// and `|T1 - u_in| <= |T1| + |u_in| <= 380 + 380 = 760` (`T_outer_1`'s
// own amplitude plus offset, doubled rather than exploiting that `u_in`
// tracks `T1` more closely at convergence -- a deliberately loose,
// never-underestimating bound) gives `|q_in| <= 0.96 * 760 ~= 730`.
// Floor <= `u * (295 + 730) ~= 1.11e-16 * 1025 ~= 1.1e-13` K -- `1e-10 /
// 1.1e-13 ~= 909`, i.e. roughly THREE orders of magnitude below a
// single period's own sharp-bound terms (~1e-10 K, the table in
// docs/user/dn-heat.md), and not itself the result of "two
// different arithmetics evaluating the same map": the committing
// evaluation (`evaluate_region_residual(..., commit=true)`) is the
// IDENTICAL compiled function for both substrates, called once on
// whatever `x` that substrate's own Newton solve produced -- any
// floating-point disagreement in the COMMITTED `u_final` traces back to
// `x_n != x_p` as floating-point numbers, which the sharp bound's own
// terminal-residual terms already measure through the SAME identity. No
// separate floor is added to the formula below; if the host's own
// observed disagreement sits near this ~1e-13 K level rather than near
// the sharp bound's own prediction, that is the round-off floor showing
// up empirically, to be reported and derived further, never tuned to.
// The round-4 bound sums every period's own terms rather than using
// only one period's, so it is never SMALLER than the single-period
// figure used in the ~909x comparison above -- the margin between this
// floor and the asserted bound only widens once summed over `kPeriods`
// periods, never narrows.

#include <cmath>
#include <vector>

namespace dn_heat {

/// The smallest station conductivity ratio dn_component.hpp's own
/// `kStripRatio` defines -- the station at which `||J^-1||_inf` is
/// maximised (see this header's own derivation above).
inline constexpr double kMinStationRatio = 0.50;

/// `||J^-1||_inf` for dn_heat's per-station affine Jacobian, exact (not
/// rounded): `max(1, 2/(1+r_i))` over `r_i` in `kStripRatio`, maximised at
/// `kMinStationRatio`.
inline constexpr double kJacobianInverseInfNorm = 2.0 / (1.0 + kMinStationRatio);

/// The SHARP agreement bound (round 4: this header's own "PER-PERIOD
/// RECURSION" section above). `final_residuals_native`/`_petsc` are
/// each solver's own TERMINAL residual FOR EVERY PERIOD
/// (`Outcome::per_period_final_residual`, in the 2-norm,
/// `DistVector::norm2()`) -- not only the last one. The two vectors
/// need not be the same length (a caller comparing runs with different
/// period counts is still summed correctly, each contributing its own
/// terms independently); summation order does not matter mathematically
/// (addition is commutative and this header makes no claim tighter than
/// IEEE double rounding, which can differ in the last bit by
/// reassociation -- never claimed to be avoided here). An empty vector
/// contributes nothing (0.0): a solver that ran zero periods
/// disagreed with nothing, so it is correct for the bound to reflect
/// that, not to refuse. No safety margin, no separate inter-Krylov
/// term: every input is a measured terminal residual and the only
/// constant is the exact `||J^-1||_inf`.
[[nodiscard]] inline double agreement_bound(
    const std::vector<double>& final_residuals_native,
    const std::vector<double>& final_residuals_petsc) {
  double sum = 0.0;
  for (const double f : final_residuals_native) sum += f;
  for (const double f : final_residuals_petsc) sum += f;
  return kJacobianInverseInfNorm * sum;
}

// ---------------------------------------------------------------------
// LEGACY (round 2's own formula) -- kept ONLY so `main.cpp`'s own check
// can print it, labelled `[agreement-bound legacy]`, alongside the sharp
// bound above, informationally: the owner's own round-3 instruction asks
// for both numbers so a host run can report observed/sharp AND
// observed/legacy. NEVER asserted against -- round 2's own diagnosis
// (this header's own "WHAT WAS WRONG" section) is why.
// ---------------------------------------------------------------------

/// The measured relative inter-solver (native-vs-PETSc GMRES/Arnoldi)
/// difference, docs/dev/newton-krylov.md §10.3 -- round 2's own
/// additional term, kept here only for the legacy formula below.
inline constexpr double kMeasuredInterSolverRelativeDifference = 3.9e-8;

/// Round 2's own formula, verbatim: scales by each solver's own stopping
/// criterion (`rtol * ||F_0||`) rather than its terminal residual, plus
/// the inter-Krylov term this round's own derivation found to be the
/// wrong kind of quantity for this purpose. See this header's own
/// top-of-file "WHAT WAS WRONG" section.
[[nodiscard]] inline double legacy_agreement_bound(
    double rtol_native, double initial_residual_norm_native, double rtol_petsc,
    double initial_residual_norm_petsc, double native_solution_norm) {
  const double rtol_term = kJacobianInverseInfNorm *
      (rtol_native * initial_residual_norm_native +
       rtol_petsc * initial_residual_norm_petsc);
  const double gmres_term =
      kMeasuredInterSolverRelativeDifference * native_solution_norm;
  return rtol_term + gmres_term;
}

}  // namespace dn_heat
