#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "fuse/num/ssp_coefficient.hpp"

/// S8 physics-kit, track "Integrators", Stage B Round 1 (docs/dev/integrators-design.md
/// Section 4.1; the owner's six Round-1 answers are recorded there under "Round 1 built").
///
/// Every coefficient below is copied from upstream/karma/src/numerics/temporal/*.C
/// (pinned sha 136f520, UPSTREAMS.lock [karma]) -- one file:line citation per factory,
/// transcribed as the literal C++ expression the vendored source itself used wherever
/// that was a clean fraction, and as the identical decimal literal otherwise. Nothing
/// here is re-derived from the literature; D-A (2026-10-27, "adapt the FEM core to
/// fuse::linalg") is read as leaving the COEFFICIENTS of a published method alone --
/// only the surrounding integrator machinery is adapted (Section 4 of the design doc).
///
/// Two corrections against the vendored metadata, both re-verified in this round by
/// re-reading the cited source lines directly (not carried over from the Stage A
/// design by reference):
///   - DOPRI45's own `_secondary_order` (rk.C's 4-argument RK() constructor sets it
///     equal to the primary order, 5, when the 3-argument overload that would set a
///     true secondary order is not used, dopri45.C:22-23) is WRONG: the literal `E`
///     coefficients (dopri45.C:25-26) are the literature's own Dormand-Prince 5(4)
///     error-estimation coefficients, whose embedded order is 4. `embedded_order` below
///     is set to the CORRECT value, 4, not the vendored (wrong) 5.
///   - DOPRI45's own `is_FSAL()` is hardcoded `false` (dopri45.h, MEASURED), even
///     though the tableau has the textbook FSAL structure: row 6 of `A` is bit-identical
///     to `B` (dopri45.C:31-32 vs 45-46) and `C[6] == 1.0` (dopri45.C:29) -- the defining
///     property of a first-same-as-last method (Hairer, Norsett & Wanner, Solving ODEs I,
///     II.5). `is_FSAL` below is set to the CORRECT value, true, not the vendored false.
///     NEW finding this round (not in the Stage A design's Section 6.1).
///
/// A THIRD finding this round, surfaced by `butcher_row_sum_defect()` below rather than
/// by reading: SDIRK332's own `C[1]` (`24.0/108.0`, sdirk332.C:29) does not equal row 1
/// of `A` summed (`-61/108 + 5/6 = 29/108`, sdirk332.C:40) -- a defect of `5/108 ~= 0.046`,
/// far above floating round-off (every OTHER row of every OTHER ERK/DIRK/SDIRK/FIRK
/// tableau here satisfies `C[i] == sum_j A[i][j]` to ~1e-16, printed in
/// test/num/butcher_tableau_test.cpp). RESOLVED 2026-10-29 against the primary source
/// (Norsett and Thomsen, BIT 24 (1984), scheme (5.4), see `sdirk332_tableau()`): `C[1]`
/// is `29/108`; the shipped tableau carries it, and the row-sum test now holds for every
/// classical tableau without exception.
///
/// A FOURTH set of findings (lead, 2026-10-29), from evaluating each tableau's linear
/// stability function R(z) = 1 + z b^T (I - z M)^{-1} 1 rather than trusting the vendored
/// `is_A_stable()`/`is_L_stable()` bodies: SDIRK332 and ROS3P are A-stable but NOT
/// L-stable (R(-inf) = -91/125 and 1 - sqrt(3)); GRK4Sn is not L-stable (R(-inf) = 1/3);
/// GRK4T is A(89.3 deg)-stable only (max |R(iy)| = 1.028) and not L-stable
/// (R(-inf) = 0.454); the vendored SDIRK34 and SDIRK45 are not A-stable at all (see their
/// factories). Every flag below is now the computed one, and
/// `ButcherTableau.DeclaredStabilityMatchesTheStabilityFunction` recomputes R(z) from the
/// shipped coefficients so a future copy error cannot survive.
///
/// GRK4A's embedded estimate (Stage A design Section 6.1) is RE-VERIFIED here, not
/// assumed: grk4a.C:71 still reads `E[i] = B[i] - B[i]` for every stage (this round's own
/// read, independent of the prior one) -- identically zero, because the vendored "d prime"
/// row (grk4a.C:60-64) is a verbatim copy of "d" (grk4a.C:55-59). `embedded_order` is left
/// `std::nullopt` for GRK4A (not the vendored category's claimed 3) so a caller asking
/// "does this method have a usable embedded error estimate" gets the honest answer; `E` is
/// still populated with the vendored (zero) values for transparency, not omitted.
///
/// A FIFTH finding, Round 2 (task item 1's own generic rooted-tree order-condition
/// checker, `fuse/num/order_conditions.hpp`): SDIRK44's vendored comment ("Fourth order
/// L-stable and stiffly accurate scheme", sdirk44.C:25) is wrong. Every order condition
/// through 3 holds to round-off; the order-4 condition `sum_i b_i c_i^3 = 1/4` fails by
/// EXACTLY `1/18` in exact rational arithmetic (`b.c^3 = 11/36`), not round-off --
/// `sdirk44_tableau()`'s own doc comment has the derivation. `order` is corrected to 3.
///
/// The SAME checker does NOT validate the Rosenbrock family's order beyond order 2: a
/// Rosenbrock/ROW method using an EXACT Jacobian reduces to the classical RK order
/// conditions on `rosenbrock_classical_form()`'s own (M, b) only up to order 2 (Hairer &
/// Wanner, Solving ODEs II, IV.7 -- the Jacobian-dependent "extra" order conditions a ROW
/// method also needs first appear at order 3); MEASURED directly, every one of ROS22
/// (order 2), GRK4A/GRK4T/GRK4Sn/ROS3P (order 3-4)/RODASP satisfies the classical order-2
/// conditions on (M, b) to round-off, but GRK4A/GRK4T/GRK4Sn/ROS3P/RODASP all fail the
/// classical order-3-or-above conditions by amounts far above round-off --
/// `OrderConditions.RosenbrockOrderTwoConditionsHoldButHigherOrderClassicalConditionsDoNot`
/// (order_conditions_test.cpp) pins this as the reason task item 3's own Rosenbrock
/// verification is a MEASURED numerical order test, not this checker, above order 2.
///
/// `gamma` is populated for the six Rosenbrock tableaus for DATA FIDELITY, in the form the
/// vendored files store (identified 2026-10-29 by recovering ROS3P's published weights
/// b = (2/3, 0, 1/3) and gamma_21 = -1 from it): it is Hairer and Wanner's implementation
/// form (Solving ODEs II, IV.7, (7.4)), scaled by gamma. With Gamma the method's own
/// lower-triangular matrix (diagonal gamma) and alpha its stage matrix, the stored
/// fields are `A[i][j] = (alpha Gamma^{-1})_{ij}`, `gamma[i][j] = (I - gamma Gamma^{-1})_{ij}`
/// off the diagonal with `gamma[i][i] = gamma`, and `B[i] = (Gamma^{-T} b)_i`; the
/// inverse map is `Gamma = ((I - gamma_offdiag) / gamma)^{-1}`, `alpha = A Gamma`,
/// `b = Gamma^T B`, and the stability function is that of the DIRK with matrix
/// alpha + Gamma. The Rosenbrock integrator (round 2) consumes the stored form directly;
/// the test recovers the classical form only to check the flags.
/// Round 1 implements only the explicit family
/// (integrator.hpp, explicit_integrator.hpp); the Rosenbrock integrator itself, and the
/// consistency condition that actually applies to a Rosenbrock tableau (which is NOT the
/// classical `C[i] = sum_j A[i][j]` row-sum relation -- see `butcher_row_sum_defect()`'s
/// own doc comment), are Round 2 scope (D-B).
namespace fuse::num {

/// Which family of Runge-Kutta/Rosenbrock method a ButcherTableau describes. Names are
/// KARMA's own RKCategory names (include/enums/runge_kutta_schemes.h:29-40), which are
/// themselves standard literature terms (Hairer & Wanner, Solving Ordinary Differential
/// Equations I/II): explicit RK (Erk), diagonally implicit RK (Dirk), singly diagonally
/// implicit RK (Sdirk), fully implicit RK with a nonsingular/singular A (FirkNonSingular/
/// FirkSingular), and Rosenbrock (linearised-per-stage) methods, each with an *Embedded
/// variant carrying a second, lower-order solution for step-size control.
enum class ButcherFamily {
  Erk,
  ErkEmbedded,
  Dirk,
  DirkEmbedded,
  Sdirk,
  SdirkEmbedded,
  FirkNonSingular,
  FirkSingular,
  Rosenbrock,
  RosenbrockEmbedded,
};

[[nodiscard]] inline const char* to_cstring(ButcherFamily f) noexcept {
  switch (f) {
    case ButcherFamily::Erk:                 return "Erk";
    case ButcherFamily::ErkEmbedded:         return "ErkEmbedded";
    case ButcherFamily::Dirk:                return "Dirk";
    case ButcherFamily::DirkEmbedded:        return "DirkEmbedded";
    case ButcherFamily::Sdirk:                return "Sdirk";
    case ButcherFamily::SdirkEmbedded:        return "SdirkEmbedded";
    case ButcherFamily::FirkNonSingular:     return "FirkNonSingular";
    case ButcherFamily::FirkSingular:        return "FirkSingular";
    case ButcherFamily::Rosenbrock:          return "Rosenbrock";
    case ButcherFamily::RosenbrockEmbedded:  return "RosenbrockEmbedded";
  }
  return "Unknown";
}

/// A Runge-Kutta or Rosenbrock method's defining coefficients: Hairer & Wanner's Butcher
/// tableau (`A`, `B`, `C`), the embedded pair's weight difference `E = B - B_hat` when one
/// exists, and Rosenbrock's own linearisation matrix `gamma` when the family needs one.
/// `name`/`family`/`stages`/`order` and the four stability predicates mirror KARMA's own
/// `RK`/`KARMATSBase` member data (upstream/karma/include/numerics/temporal/rk.h,
/// karma_ts_base.h) -- read, not redesigned: Section 1.2 of the Stage A design is this
/// struct's own source for every field except `gamma` (new, Section 1.2 extended per this
/// header's own top comment, for Rosenbrock data fidelity).
struct ButcherTableau {
  std::string name;
  ButcherFamily family = ButcherFamily::Erk;
  int stages = 0;
  int order = 0;
  /// nullopt: no usable embedded error estimate (FE, Kutta22/44, SDIRK22/23/33/34/44/45,
  /// ROS22 -- genuinely non-embedded -- and GRK4A, whose vendored E is identically zero,
  /// see this header's own top comment).
  std::optional<int> embedded_order;
  std::vector<std::vector<double>> A;
  std::vector<double> B;
  std::vector<double> C;
  /// B - B_hat, present iff the vendored source defines an `E` array at all (even when,
  /// as for GRK4A, every entry is zero -- `has_embedded_estimate()` is the honest query).
  std::optional<std::vector<double>> E;
  /// Rosenbrock's linearisation matrix (`G` in the vendored source). Round 2 scope; carried
  /// here for data fidelity only (this header's own top comment).
  std::optional<std::vector<std::vector<double>>> gamma;
  bool is_A_stable = false;
  bool is_L_stable = false;
  bool is_S_stable = false;
  bool is_FSAL = false;
  /// Round 2 (task item 4), owner decision (b): the strong-stability-preserving (SSP,
  /// "total variation diminishing") coefficient, `nullopt` for every non-SSP tableau.
  /// Carried as metadata exactly like `is_A_stable` -- COMPUTED (ssp_coefficient.hpp's
  /// own `ssp_coefficient_from_shu_osher()`), never a copied published number, per the
  /// task's own instruction ("must be REPRODUCED by the computation, not copied").
  std::optional<double> ssp_coefficient;

  /// Round 3 (task items 1-3): the Kennedy & Carpenter dense-output table, `b*_ij`
  /// (NASA TM-2001-211038 Appendix D, eq. 54 of the TM's own main text: `b_i(theta) =
  /// sum_{j=1}^{p*} b*_ij * theta^j`). `dense_output[i]` is stage i's own row of `p*`
  /// coefficients (j=1..p*, in order); `dense_output_order` is `p*`. `nullopt` for
  /// every tableau this project ships that does not publish one of its own (every
  /// classical/SDIRK/Rosenbrock tableau, and the explicit family, all fall back to
  /// `CubicHermiteExtension`, continuous_extension.hpp's own top comment).
  std::optional<std::vector<std::vector<double>>> dense_output;
  std::optional<int> dense_output_order;

  /// The honest query Section 4.2 of the design doc (EmbeddedController) and
  /// embedded_controller.hpp's own refusals are built on: true only when both an
  /// embedded order AND a (not-necessarily-nonzero, but AT LEAST present) `E` are on
  /// record. GRK4A therefore reports false here, by construction, not by a caller
  /// remembering to special-case it.
  [[nodiscard]] bool has_embedded_estimate() const noexcept {
    return embedded_order.has_value() && E.has_value();
  }

  /// The SAME kind of honest query for Round 3's own dense-output addition: true only
  /// when both the table AND its order are on record.
  [[nodiscard]] bool has_dense_output() const noexcept {
    return dense_output.has_value() && dense_output_order.has_value();
  }
};

/// `max_i |C[i] - sum_j A[i][j]|`: the Butcher row-sum consistency condition every
/// classical (Erk/Dirk/Sdirk/Firk*) method must satisfy (Hairer, Norsett & Wanner,
/// Solving Ordinary Differential Equations I, Theorem II.1.5's own `c_i = sum_j a_ij`
/// convention) -- a nonzero value well above floating round-off means the tableau cannot
/// be consistent, let alone of its claimed order, independent of ever running a single
/// step. Returns 0 for a tableau with no stages.
///
/// DOES NOT APPLY to the Rosenbrock family (gamma != nullopt): a Rosenbrock method's own
/// consistency relation involves BOTH `A` and `gamma` together (Hairer & Wanner, Solving
/// ODEs II, IV.7, equations (7.3)-(7.5)), not `A` alone -- computing this function on
/// ROS22 or GRK4A returns a large, MEANINGLESS number (the vendored `C` there is the
/// "alpha" stage-time vector of the ROW formulation, not a row-sum of the plain `A` the
/// way the ERK/DIRK/SDIRK/FIRK families use it). Callers check `family` first; Round 2
/// derives the Rosenbrock-specific condition when the Rosenbrock integrator is built.
[[nodiscard]] inline double butcher_row_sum_defect(const ButcherTableau& t) noexcept {
  double worst = 0.0;
  for (int i = 0; i < t.stages; ++i) {
    double row_sum = 0.0;
    for (int j = 0; j < t.stages; ++j) row_sum += t.A[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
    const double defect = t.C[static_cast<std::size_t>(i)] - row_sum;
    worst = std::max(worst, defect < 0.0 ? -defect : defect);
  }
  return worst;
}

/// `|sum(B) - 1|`: the order-1 consistency condition every method claiming order >= 1
/// must satisfy (same theorem as `butcher_row_sum_defect`). Applies to the same families
/// that function does, for the same reason (the Rosenbrock `B` here is "d", which the
/// vendored `RosenbrockIntegrator::core_integrator()` combines with `k_i` that are
/// themselves already O(h)-scaled by the stage's own linear solve -- a different relation
/// than the plain weighted sum this function checks).
[[nodiscard]] inline double butcher_weight_sum_defect(const ButcherTableau& t) noexcept {
  double sum = 0.0;
  for (double b : t.B) sum += b;
  return sum - 1.0 < 0.0 ? 1.0 - sum : sum - 1.0;
}

/// The classical (M, b) pair -- `M = alpha + Gamma`, `b` unchanged by the inverse map
/// -- whose stability function AND whose Butcher order conditions are identical to a
/// Rosenbrock tableau's own, PROVIDED the Jacobian used is exact (Hairer & Wanner,
/// Solving ODEs II, IV.7: a ROW method built from an exact Jacobian is order-equivalent
/// to the DIRK with this same `(M, b)`; a method that only approximates the Jacobian,
/// a "W-method", needs additional conditions this header does not check, out of this
/// track's scope -- every Rosenbrock tableau in `all_karma_tableaus()` is used with an
/// exact Jacobian action, round 2's own `RosenbrockIntegrator`). This is the SAME
/// recovery the stability test (test/num/butcher_tableau_test.cpp's own `stability_data`)
/// performs locally; it is reproduced here, not shared by #include, because the test
/// file's own copy predates this function and changing a passing test's own internals
/// is out of this round's scope -- the two are two independent implementations of the
/// same published map, which is exactly why `ButcherTableau.DeclaredStabilityMatchesThe
/// StabilityFunction` and `OrderConditions.RosenbrockTableausSatisfyTheirDeclaredOrder`
/// (order_conditions.hpp's own test) agreeing on every Rosenbrock tableau is independent
/// confirmation, not a single formula checked against itself twice.
///
/// Returns `{A, B}` UNCHANGED for a tableau with no `gamma` (every classical family):
/// those already use `A`/`B` directly for both stability and order conditions.
struct ClassicalForm {
  std::vector<std::vector<double>> M;
  std::vector<double> b;
};

[[nodiscard]] inline ClassicalForm rosenbrock_classical_form(const ButcherTableau& t) {
  if (!t.gamma.has_value()) return ClassicalForm{t.A, t.B};
  const std::size_t n = static_cast<std::size_t>(t.stages);
  const double g = (*t.gamma)[0][0];
  // Ginv = (I - gamma_offdiag) / g, lower triangular with unit-scaled diagonal 1/g.
  std::vector<std::vector<double>> ginv(n, std::vector<double>(n, 0.0));
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j <= i; ++j) {
      ginv[i][j] = ((i == j) ? 1.0 : -(*t.gamma)[i][j]) / g;
    }
  }
  // Gamma = Ginv^{-1} (forward substitution; Ginv is lower-triangular).
  std::vector<std::vector<double>> gam(n, std::vector<double>(n, 0.0));
  for (std::size_t j = 0; j < n; ++j) {
    gam[j][j] = 1.0 / ginv[j][j];
    for (std::size_t i = j + 1; i < n; ++i) {
      double s = 0.0;
      for (std::size_t k = j; k < i; ++k) s += ginv[i][k] * gam[k][j];
      gam[i][j] = -s / ginv[i][i];
    }
  }
  std::vector<std::vector<double>> m(n, std::vector<double>(n, 0.0));
  std::vector<double> b(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      double a = 0.0;
      for (std::size_t k = 0; k < n; ++k) a += t.A[i][k] * gam[k][j];
      m[i][j] = a + gam[i][j];
      b[j] += gam[i][j] * t.B[i];
    }
  }
  return ClassicalForm{m, b};
}

// ===========================================================================
// Factories, one per KARMA RKType (include/enums/runge_kutta_schemes.h:42-64).
// Round 1 implements the first four (FE, Kutta22, Kutta44, DOPRI45) as the explicit
// family (explicit_integrator.hpp); the remaining 18 are data only in this round.
// ===========================================================================


/// FE (forward_euler.C:22-28).
[[nodiscard]] inline ButcherTableau forward_euler_tableau() {
  ButcherTableau t;
  t.name = "FE";
  t.family = ButcherFamily::Erk;
  t.stages = 1;
  t.order = 1;
  t.embedded_order = std::nullopt;
  t.A = std::vector<std::vector<double>>{
      {0.0}
    };
  t.B = std::vector<double>{1.0};
  t.C = std::vector<double>{0.0};
  t.E = std::nullopt;
  t.gamma = std::nullopt;
  t.is_A_stable = false;
  t.is_L_stable = false;
  t.is_S_stable = false;
  t.is_FSAL = false;
  return t;
}


/// KUTTA22 (kutta22.C:22-35).
[[nodiscard]] inline ButcherTableau kutta22_tableau() {
  ButcherTableau t;
  t.name = "KUTTA22";
  t.family = ButcherFamily::Erk;
  t.stages = 2;
  t.order = 2;
  t.embedded_order = std::nullopt;
  t.A = std::vector<std::vector<double>>{
      {0.0, 0.0},
      {0.6666666666666666, 0.0}
    };
  t.B = std::vector<double>{0.25, 0.75};
  t.C = std::vector<double>{0.0, 0.6666666666666666};
  t.E = std::nullopt;
  t.gamma = std::nullopt;
  t.is_A_stable = false;
  t.is_L_stable = false;
  t.is_S_stable = false;
  t.is_FSAL = false;
  return t;
}


/// KUTTA44 (kutta44.C:22-37).
[[nodiscard]] inline ButcherTableau kutta44_tableau() {
  ButcherTableau t;
  t.name = "KUTTA44";
  t.family = ButcherFamily::Erk;
  t.stages = 4;
  t.order = 4;
  t.embedded_order = std::nullopt;
  t.A = std::vector<std::vector<double>>{
      {0.0, 0.0, 0.0, 0.0},
      {0.3333333333333333, 0.0, 0.0, 0.0},
      {-0.3333333333333333, 1.0, 0.0, 0.0},
      {1.0, -1.0, 1.0, 0.0}
    };
  t.B = std::vector<double>{0.125, 0.375, 0.375, 0.125};
  t.C = std::vector<double>{0.0, 0.3333333333333333, 0.6666666666666666, 1.0};
  t.E = std::nullopt;
  t.gamma = std::nullopt;
  t.is_A_stable = false;
  t.is_L_stable = false;
  t.is_S_stable = false;
  t.is_FSAL = false;
  return t;
}


/// DOPRI45 (dopri45.C:22-47 (embedded_order corrected to 4, is_FSAL corrected to true: see header comment)).
[[nodiscard]] inline ButcherTableau dopri45_tableau() {
  ButcherTableau t;
  t.name = "DOPRI45";
  t.family = ButcherFamily::ErkEmbedded;
  t.stages = 7;
  t.order = 5;
  t.embedded_order = 4;
  t.A = std::vector<std::vector<double>>{
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0.075, 0.225, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0.9777777777777777, -3.7333333333333334, 3.5555555555555554, 0.0, 0.0, 0.0, 0.0},
      {2.9525986892242035, -11.595793324188385, 9.822892851699436, -0.2908093278463649, 0.0, 0.0, 0.0},
      {2.8462752525252526, -10.757575757575758, 8.906422717743473, 0.2784090909090909, -0.2735313036020583, 0.0, 0.0},
      {0.09114583333333333, 0.0, 0.44923629829290207, 0.6510416666666666, -0.322376179245283, 0.13095238095238096, 0.0}
    };
  t.B = std::vector<double>{0.09114583333333333, 0.0, 0.44923629829290207, 0.6510416666666666, -0.322376179245283, 0.13095238095238096, 0.0};
  t.C = std::vector<double>{0.0, 0.2, 0.3, 0.8, 0.8888888888888888, 1.0, 1.0};
  t.E = std::vector<double>{0.0012326388888888888, 0.0, -0.0042527702905061394, 0.03697916666666667, -0.05086379716981132, 0.0419047619047619, -0.025};
  t.gamma = std::nullopt;
  t.is_A_stable = false;
  t.is_L_stable = false;
  t.is_S_stable = false;
  t.is_FSAL = true;
  return t;
}


/// BE (backward_euler.C:22-32).
[[nodiscard]] inline ButcherTableau backward_euler_tableau() {
  ButcherTableau t;
  t.name = "BE";
  t.family = ButcherFamily::FirkNonSingular;
  t.stages = 1;
  t.order = 1;
  t.embedded_order = std::nullopt;
  t.A = std::vector<std::vector<double>>{
      {1.0}
    };
  t.B = std::vector<double>{1.0};
  t.C = std::vector<double>{1.0};
  t.E = std::nullopt;
  t.gamma = std::nullopt;
  t.is_A_stable = true;
  t.is_L_stable = true;
  t.is_S_stable = false;
  t.is_FSAL = true;
  return t;
}


/// CN (crank_nicolson.C:22-48 (#else branch; #ifdef IMPLICIT_MIDPOINT branch is dead, never defined)).
[[nodiscard]] inline ButcherTableau crank_nicolson_tableau() {
  ButcherTableau t;
  t.name = "CN";
  t.family = ButcherFamily::FirkSingular;
  t.stages = 2;
  t.order = 2;
  t.embedded_order = std::nullopt;
  t.A = std::vector<std::vector<double>>{
      {0.0, 0.0},
      {0.5, 0.5}
    };
  t.B = std::vector<double>{0.5, 0.5};
  t.C = std::vector<double>{0.0, 1.0};
  t.E = std::nullopt;
  t.gamma = std::nullopt;
  t.is_A_stable = true;
  t.is_L_stable = false;
  t.is_S_stable = false;
  t.is_FSAL = true;
  return t;
}


/// SDIRK22 (sdirk22.C:22-65, gamma=1-1/sqrt(2)).
[[nodiscard]] inline ButcherTableau sdirk22_tableau() {
  ButcherTableau t;
  t.name = "SDIRK22";
  t.family = ButcherFamily::Sdirk;
  t.stages = 2;
  t.order = 2;
  t.embedded_order = std::nullopt;
  t.A = std::vector<std::vector<double>>{
      {0.29289321881345254, 0.0},
      {0.7071067811865475, 0.29289321881345254}
    };
  t.B = std::vector<double>{0.7071067811865475, 0.29289321881345254};
  t.C = std::vector<double>{0.29289321881345254, 1.0};
  t.E = std::nullopt;
  t.gamma = std::nullopt;
  t.is_A_stable = true;
  t.is_L_stable = true;
  t.is_S_stable = true;
  t.is_FSAL = true;
  return t;
}


/// SDIRK23 (sdirk23.C:22-38, gamma=(3+sqrt(3))/6).
[[nodiscard]] inline ButcherTableau sdirk23_tableau() {
  ButcherTableau t;
  t.name = "SDIRK23";
  t.family = ButcherFamily::Sdirk;
  t.stages = 2;
  t.order = 3;
  t.embedded_order = std::nullopt;
  t.A = std::vector<std::vector<double>>{
      {0.7886751345948128, 0.0},
      {-0.5773502691896255, 0.7886751345948128}
    };
  t.B = std::vector<double>{0.5, 0.5};
  t.C = std::vector<double>{0.7886751345948128, 0.21132486540518725};
  t.E = std::nullopt;
  t.gamma = std::nullopt;
  t.is_A_stable = true;
  t.is_L_stable = false;
  t.is_S_stable = true;
  t.is_FSAL = false;
  return t;
}


/// SDIRK34 (sdirk34.C:22-56): the 3-stage, order-4 SDIRK of Crouzeix and Norsett, with gamma
/// the LARGEST root 1.0685790213016289 of x^3 - 3/2 x^2 + 1/2 x - 1/24 (the other two roots,
/// 0.12888640051572 and 0.30253457818265, give |R(-inf)| = 9.82 and 1.45: bounded stability
/// regions, not A-stable). The vendored file uses the smallest root and declares the method
/// A-stable (sdirk34.h:60); owner decision 2026-10-29: ship the A-stable method, R(-inf) =
/// -0.6304, and pin the vendored root's instability in a test rather than ship it under a
/// second name. Coefficients: a21 = 1/2 - gamma, a31 = 2 gamma, a32 = 1 - 4 gamma,
/// b1 = b3 = q/24, b2 = 1 - q/12 with q = 1/(1/2 - gamma)^2, c = (gamma, 1/2, 1 - gamma);
/// c3 < 0 is a property of the method, not an error.
[[nodiscard]] inline ButcherTableau sdirk34_tableau() {
  ButcherTableau t;
  t.name = "SDIRK34";
  t.family = ButcherFamily::Sdirk;
  t.stages = 3;
  t.order = 4;
  t.embedded_order = std::nullopt;
  t.A = std::vector<std::vector<double>>{
      {1.0685790213016289, 0.0, 0.0},
      {-0.5685790213016289, 1.0685790213016289, 0.0},
      {2.1371580426032577, -3.2743160852065154, 1.0685790213016289}
    };
  t.B = std::vector<double>{0.12888640051572042, 0.7422271989685592, 0.12888640051572042};
  t.C = std::vector<double>{1.0685790213016289, 0.5, -0.06857902130162881};
  t.E = std::nullopt;
  t.gamma = std::nullopt;
  t.is_A_stable = true;
  t.is_L_stable = false;
  t.is_S_stable = true;
  t.is_FSAL = false;
  return t;
}


/// SDIRK332 (sdirk332.C:22-42, gamma=5/6). Primary source: S. P. Norsett and P. G. Thomsen,
/// "Embedded SDIRK-methods of basic order three", BIT 24 (1984) 634-646, scheme (5.4) ("based
/// on (3.15) using local extrapolation"): the propagated weights are the paper's third-order
/// e-row (26/61, 324/671, 1/11) and the error estimate subtracts its second-order b-row
/// (25/61, 36/61, 0). Two corrections against the vendored file, both verified 2026-10-29:
/// C[1] = 29/108 (the paper's value and the row sum of A; the vendored 24/108 is a
/// transcription error under which even the second-order condition b.c = 1/2 fails), and
/// is_L_stable = false (R(z) -> -91/125 as z -> -inf; the method is A-stable, as the paper
/// states, not L-stable as sdirk332.h:66 claims).
[[nodiscard]] inline ButcherTableau sdirk332_tableau() {
  ButcherTableau t;
  t.name = "SDIRK332";
  t.family = ButcherFamily::SdirkEmbedded;
  t.stages = 3;
  t.order = 3;
  t.embedded_order = 2;
  t.A = std::vector<std::vector<double>>{
      {0.8333333333333334, 0.0, 0.0},
      {-0.5648148148148148, 0.8333333333333334, 0.0},
      {-0.12568306010928962, -0.5409836065573771, 0.8333333333333334}
    };
  t.B = std::vector<double>{0.4262295081967213, 0.4828614008941878, 0.09090909090909091};
  t.C = std::vector<double>{0.8333333333333334, 0.26851851851851855, 0.16666666666666663};
  t.E = std::vector<double>{0.016393442622950782, -0.10730253353204172, 0.09090909090909091};
  t.gamma = std::nullopt;
  t.is_A_stable = true;
  t.is_L_stable = false;
  t.is_S_stable = true;
  t.is_FSAL = false;
  return t;
}


/// SDIRK543 (sdirk543.C:22-43, gamma=0.25).
[[nodiscard]] inline ButcherTableau sdirk543_tableau() {
  ButcherTableau t;
  t.name = "SDIRK543";
  t.family = ButcherFamily::SdirkEmbedded;
  t.stages = 5;
  t.order = 4;
  t.embedded_order = 3;
  t.A = std::vector<std::vector<double>>{
      {0.25, 0.0, 0.0, 0.0, 0.0},
      {0.5, 0.25, 0.0, 0.0, 0.0},
      {0.34, -0.04, 0.25, 0.0, 0.0},
      {0.2727941176470588, -0.05036764705882353, 0.027573529411764705, 0.25, 0.0},
      {1.0416666666666667, -1.0208333333333333, 7.8125, -7.083333333333333, 0.25}
    };
  t.B = std::vector<double>{1.0416666666666667, -1.0208333333333333, 7.8125, -7.083333333333333, 0.25};
  t.C = std::vector<double>{0.25, 0.75, 0.55, 0.5, 1.0};
  t.E = std::vector<double>{-0.1875, -0.84375, 0.78125, 0.0, 0.25};
  t.gamma = std::nullopt;
  t.is_A_stable = true;
  t.is_L_stable = true;
  t.is_S_stable = true;
  t.is_FSAL = true;
  return t;
}


/// SDIRK643 (sdirk643.C:22-46, gamma=0.25, ESDIRK (A[0][0]=0)).
[[nodiscard]] inline ButcherTableau sdirk643_tableau() {
  ButcherTableau t;
  t.name = "SDIRK643";
  t.family = ButcherFamily::SdirkEmbedded;
  t.stages = 6;
  t.order = 4;
  t.embedded_order = 3;
  t.A = std::vector<std::vector<double>>{
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0.25, 0.25, 0.0, 0.0, 0.0, 0.0},
      {0.137776, -0.055776, 0.25, 0.0, 0.0, 0.0},
      {0.14463686602698217, -0.22393190761334475, 0.4492950415863626, 0.25, 0.0, 0.0},
      {0.09825878328356477, -0.5915442428196704, 0.8101210538282996, 0.283164405707806, 0.25, 0.0},
      {0.15791629516167136, 0.0, 0.18675894052400077, 0.6805652953093346, -0.27524053099500667, 0.25}
    };
  t.B = std::vector<double>{0.15791629516167136, 0.0, 0.18675894052400077, 0.6805652953093346, -0.27524053099500667, 0.25};
  t.C = std::vector<double>{0.0, 0.5, 0.332, 0.62, 0.85, 1.0};
  t.E = std::vector<double>{0.0032044943984591845, 0.0, -0.002446251136679456, -0.021480075919587227, 0.04394686806857245, -0.02322503541076487};
  t.gamma = std::nullopt;
  t.is_A_stable = true;
  t.is_L_stable = false;
  t.is_S_stable = false;
  t.is_FSAL = true;
  return t;
}


/// TRBE (trbe.C:22-39).
[[nodiscard]] inline ButcherTableau trbe_tableau() {
  ButcherTableau t;
  t.name = "TRBE";
  t.family = ButcherFamily::DirkEmbedded;
  t.stages = 2;
  t.order = 2;
  t.embedded_order = 1;
  t.A = std::vector<std::vector<double>>{
      {0.0, 0.0},
      {0.5, 0.5}
    };
  t.B = std::vector<double>{0.5, 0.5};
  t.C = std::vector<double>{0.0, 1.0};
  t.E = std::vector<double>{0.5, -0.5};
  t.gamma = std::nullopt;
  t.is_A_stable = true;
  t.is_L_stable = false;
  t.is_S_stable = false;
  t.is_FSAL = true;
  return t;
}


/// SDIRK33 (sdirk33.C:22-60, gamma=0.43586652150845899941601945119356).
[[nodiscard]] inline ButcherTableau sdirk33_tableau() {
  ButcherTableau t;
  t.name = "SDIRK33";
  t.family = ButcherFamily::Sdirk;
  t.stages = 3;
  t.order = 3;
  t.embedded_order = std::nullopt;
  t.A = std::vector<std::vector<double>>{
      {0.435866521508459, 0.0, 0.0},
      {0.2820667392457705, 0.435866521508459, 0.0},
      {1.20849664917601, -0.6443631706844692, 0.435866521508459}
    };
  t.B = std::vector<double>{1.20849664917601, -0.6443631706844692, 0.435866521508459};
  t.C = std::vector<double>{0.435866521508459, 0.7179332607542295, 1.0};
  t.E = std::nullopt;
  t.gamma = std::nullopt;
  t.is_A_stable = true;
  t.is_L_stable = true;
  t.is_S_stable = true;
  t.is_FSAL = true;
  return t;
}


/// SDIRK44 (sdirk44.C:22-39, "Fourth order L-stable and stiffly accurate scheme" per the
/// vendored file's own comment, line 25). FINDING (2026-10-30, order_conditions.hpp's own
/// generic rooted-tree checker, task item 1): the order-4 condition `sum_i b_i c_i^3 = 1/4`
/// fails by EXACTLY `1/18` (`b.c^3 = 11/36`, exact rational arithmetic, not round-off --
/// every condition through order 3 holds to round-off, order_conditions_test.cpp's own
/// `OrderConditions.Sdirk44sDeclaredOrderIsCorrectedFromFourToThree` pins both numbers). The vendored
/// A/B/C are carried byte-identical (there is no transcription ambiguity here, unlike
/// SDIRK332's C[1] -- every coefficient is already consistent with every lower-order
/// condition; only the metadata field claiming order 4 is wrong); `order` below is the
/// CORRECTED value, 3, the same kind of metadata-only correction this round already made
/// to `is_A_stable`/`is_L_stable` for six other tableaus from direct computation, not a
/// coefficient change.
[[nodiscard]] inline ButcherTableau sdirk44_tableau() {
  ButcherTableau t;
  t.name = "SDIRK44";
  t.family = ButcherFamily::Sdirk;
  t.stages = 4;
  t.order = 3;
  t.embedded_order = std::nullopt;
  t.A = std::vector<std::vector<double>>{
      {0.5, 0.0, 0.0, 0.0},
      {0.16666666666666666, 0.5, 0.0, 0.0},
      {-0.5, 0.5, 0.5, 0.0},
      {1.5, -1.5, 0.5, 0.5}
    };
  t.B = std::vector<double>{1.5, -1.5, 0.5, 0.5};
  t.C = std::vector<double>{0.5, 0.6666666666666666, 0.5, 1.0};
  t.E = std::nullopt;
  t.gamma = std::nullopt;
  t.is_A_stable = true;
  t.is_L_stable = true;
  t.is_S_stable = true;
  t.is_FSAL = true;
  return t;
}


/// SDIRK45 (sdirk45.C:22-40) is NOT ported (owner decision 2026-10-29). The vendored tableau
/// is not singly diagonal (diagonal 0, 1/8, 3/20, 0: an explicit first AND last stage with two
/// distinct implicit diagonals), and its stability function is unbounded (R(z) -> inf as
/// z -> -inf) although sdirk45.h:60 declares it A-stable; the order conditions through 4 and
/// the linear order-5 condition do hold. A method with a bounded stability region has no use
/// in the implicit kit.


/// ROS22 (ros22.C:22-65).
[[nodiscard]] inline ButcherTableau ros22_tableau() {
  ButcherTableau t;
  t.name = "ROS22";
  t.family = ButcherFamily::Rosenbrock;
  t.stages = 2;
  t.order = 2;
  t.embedded_order = std::nullopt;
  t.A = std::vector<std::vector<double>>{
      {0.0, 0.0},
      {3.4142135623730945, 0.0}
    };
  t.B = std::vector<double>{5.121320343559642, 1.7071067811865472};
  t.C = std::vector<double>{0.0, 1.0};
  t.E = std::nullopt;
  t.gamma = std::vector<std::vector<double>>{
      {0.29289321881345254, 0.0},
      {-2.0, 0.29289321881345254}
    };
  t.is_A_stable = true;
  t.is_L_stable = true;
  t.is_S_stable = true;
  t.is_FSAL = false;
  return t;
}


/// GRK4A (grk4a.C:22-82; embedded_order left None: E[i]=B[i]-B[i] is identically zero (the vendored 'd prime' row, grk4a.C:60-64, is a verbatim copy of 'd', grk4a.C:55-59, not Kaps & Rentrop's actual lower-order coefficients) -- RE-VERIFIED by re-reading grk4a.C in this round, same MEASURED finding as the Stage A design (Section 6.1)).
[[nodiscard]] inline ButcherTableau grk4a_tableau() {
  ButcherTableau t;
  t.name = "GRK4A";
  t.family = ButcherFamily::RosenbrockEmbedded;
  t.stages = 4;
  t.order = 4;
  t.embedded_order = std::nullopt;
  t.A = std::vector<std::vector<double>>{
      {0.0, 0.0, 0.0, 0.0},
      {1.1088607594936706, 0.0, 0.0, 0.0},
      {2.3770852619819602, 0.1850114988898733, 0.0, 0.0},
      {2.3770852619819602, 0.1850114988898733, 0.0, 0.0}
    };
  t.B = std::vector<double>{1.8456832404082615, 0.13697968943635339, 0.7129097783295112, 0.6329113924050633};
  t.C = std::vector<double>{0.0, 0.438, 0.8699999999995001, 0.8699999999995001};
  t.E = std::vector<double>{0.0, 0.0, 0.0, 0.0};
  t.gamma = std::vector<std::vector<double>>{
      {0.395, 0.0, 0.0, 0.0},
      {-1.9434744189468354, 0.395, 0.0, 0.0},
      {0.41695753098949334, 1.323967820729114, 0.395, 0.0},
      {1.5195132577836896, 1.3537081503017017, -0.8541514952582279, 0.395}
    };
  t.is_A_stable = true;
  t.is_L_stable = false;
  t.is_S_stable = false;
  t.is_FSAL = false;
  return t;
}


/// GRK4T (grk4t.C:22-86). Kaps and Rentrop 1979; A(89.3 deg)-stable, not A-stable, R(-inf) = 0.454:
/// grk4t.h declares A- and L-stable, both corrected here (lead, 2026-10-29, from R(z)).
[[nodiscard]] inline ButcherTableau grk4t_tableau() {
  ButcherTableau t;
  t.name = "GRK4T";
  t.family = ButcherFamily::RosenbrockEmbedded;
  t.stages = 4;
  t.order = 4;
  t.embedded_order = 3;
  t.A = std::vector<std::vector<double>>{
      {0.0, 0.0, 0.0, 0.0},
      {2.0, 0.0, 0.0, 0.0},
      {4.52470820736, 4.1635287886, 0.0, 0.0},
      {4.52470820736, 4.1635287886, 0.0, 0.0}
    };
  t.B = std::vector<double>{3.95750374663, 4.62489238836, 0.617477263873, 1.282612945268};
  t.C = std::vector<double>{0.0, 0.462, 0.880208333333, 0.880208333333};
  t.E = std::vector<double>{-2.30215540292, -3.0736344853899995, 0.873280801802, 1.282612945268};
  t.gamma = std::vector<std::vector<double>>{
      {0.231, 0.0, 0.0, 0.0},
      {-1.1715570032558702, 0.231, 0.0, 0.0},
      {1.3906552803181502, 0.036902408159463, 0.231, 0.0},
      {-0.428815375914387, -1.96474297824189, -0.48142135642062006, 0.231}
    };
  t.is_A_stable = false;
  t.is_L_stable = false;
  t.is_S_stable = false;
  t.is_FSAL = false;
  return t;
}


/// GRK4Sn (grk4sn.C:22-86). A-stable with R(-inf) = 1/3, not L-stable as grk4sn.h declares;
/// corrected (lead, 2026-10-29, from R(z)).
[[nodiscard]] inline ButcherTableau grk4sn_tableau() {
  ButcherTableau t;
  t.name = "GRK4Sn";
  t.family = ButcherFamily::RosenbrockEmbedded;
  t.stages = 4;
  t.order = 4;
  t.embedded_order = 3;
  t.A = std::vector<std::vector<double>>{
      {0.0, 0.0, 0.0, 0.0},
      {2.0, 0.0, 0.0, 0.0},
      {1.92, 0.24, 0.0, 0.0},
      {1.92, 0.24, 0.0, 0.0}
    };
  t.B = std::vector<double>{2.111111111111111, 0.5, 0.23148148148148148, 1.1574074074074074};
  t.C = std::vector<double>{0.0, 1.0, 0.6, 0.6};
  t.E = std::vector<double>{0.3148148148148149, 0.19444444444444442, 0.0, 1.1574074074074074};
  t.gamma = std::vector<std::vector<double>>{
      {0.5, 0.0, 0.0, 0.0},
      {-4.0, 0.5, 0.0, 0.0},
      {7.44, 1.2, 0.5, 0.0},
      {-0.448, -0.216, -0.2, 0.5}
    };
  t.is_A_stable = true;
  t.is_L_stable = false;
  t.is_S_stable = false;
  t.is_FSAL = false;
  return t;
}


/// ROS3P (ros3p.C:22-75). Lang and Verwer 2001: A-stable with R(-inf) = 1 - sqrt(3) = -0.732, by
/// design (the order-3 gamma is not the L-stable root); ros3p.C's comment and ros3p.h declare
/// L-stable, corrected here (lead, 2026-10-29, from R(z)).
[[nodiscard]] inline ButcherTableau ros3p_tableau() {
  ButcherTableau t;
  t.name = "ROS3P";
  t.family = ButcherFamily::RosenbrockEmbedded;
  t.stages = 3;
  t.order = 3;
  t.embedded_order = 2;
  t.A = std::vector<std::vector<double>>{
      {0.0, 0.0, 0.0},
      {1.267949192431123, 0.0, 0.0},
      {1.267949192431123, 0.0, 0.0}
    };
  t.B = std::vector<double>{2.0, 0.5773502691896258, 0.4226497308103742};
  t.C = std::vector<double>{0.0, 1.0, 1.0};
  t.E = std::vector<double>{-0.1132486540518709, -0.42264973081037416, 0.0};
  t.gamma = std::vector<std::vector<double>>{
      {0.7886751345948129, 0.0, 0.0},
      {-1.2679491924311224, 0.7886751345948129, 0.0},
      {-2.7320508075688776, -1.3660254037844384, 0.7886751345948129}
    };
  t.is_A_stable = true;
  t.is_L_stable = false;
  t.is_S_stable = true;
  t.is_FSAL = false;
  return t;
}


/// RODASP (rodasp.C:22-99).
[[nodiscard]] inline ButcherTableau rodasp_tableau() {
  ButcherTableau t;
  t.name = "RODASP";
  t.family = ButcherFamily::RosenbrockEmbedded;
  t.stages = 6;
  t.order = 4;
  t.embedded_order = 3;
  t.A = std::vector<std::vector<double>>{
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {3.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {1.831036793486759, 0.4955183967433795, 0.0, 0.0, 0.0, 0.0},
      {2.304376582692669, -0.05249275245743001, -1.176798761832782, 0.0, 0.0, 0.0},
      {-7.170454962423025, -4.741636671481786, -16.31002631330971, -1.062004044111401, 0.0, 0.0},
      {-7.170454962423025, -4.741636671481785, -16.31002631330971, -1.062004044111401, 1.0, 0.0}
    };
  t.B = std::vector<double>{-7.170454962423026, -4.741636671481786, -16.31002631330971, -1.062004044111401, 1.0, 1.0};
  t.C = std::vector<double>{0.0, 0.75, 0.21, 0.63, 1.0, 1.0};
  t.E = std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 1.0};
  t.gamma = std::vector<std::vector<double>>{
      {0.25, 0.0, 0.0, 0.0, 0.0, 0.0},
      {-3.0, 0.25, 0.0, 0.0, 0.0, 0.0},
      {-2.1979487934867588, -0.5519663967433796, 0.25, 0.0, 0.0, 0.0},
      {2.7044826421428825, 1.6950676528570665, 4.883714861606025, 0.25, 0.0, 0.0},
      {8.547737516874193, 3.8741778843149075, 13.686902189910326, 3.540013480371335, 0.25, 0.0},
      {8.656514577326332, 3.8252124402861827, 14.249888946656668, 4.602017524482737, -1.4285714285714293, 0.25}
    };
  t.is_A_stable = true;
  t.is_L_stable = true;
  t.is_S_stable = true;
  t.is_FSAL = false;
  return t;
}

// ===========================================================================
// Round 2 additions (task items 2 and 4): NOT from KARMA (no upstream/karma file:line
// citation applies), cited to their own published source instead, each entry copied
// VERBATIM from a source that states it is correcting or reproducing the original
// paper (both ARKode's own Butcher-table reference and a PETSc TS source comment agree
// digit-for-digit on every rational below, which is why these are entered as exact
// fractions rather than the paper's own decimal truncation). Kept OUT of
// `all_karma_tableaus()` (that function's own name and every existing test asserting
// its size, e.g. `ButcherTableau.StageCountsAndOrdersMatchTheStageAInventory`'s own
// `EXPECT_EQ(..., 21u)`, are about the 21 VENDORED tableaus specifically) -- see
// `all_round2_tableaus()` below for the new ones as a group.
// ===========================================================================

/// ESDIRK3(2)4L[2]SA (Kennedy & Carpenter, "Additive Runge-Kutta schemes for
/// convection-diffusion-reaction equations", Appl. Numer. Math. 44 (2003) 139-181,
/// the implicit half of their ARK3(2)4L[2]SA additive pair -- owner decision (d),
/// 2026-10-29, task item 2's own first-named pair). Rational coefficients entered
/// exactly as published (cross-checked against two independent secondary sources that
/// reproduce them digit-for-digit: SUNDIALS ARKode's own Butcher-table reference,
/// sundials.readthedocs.io/en/latest/arkode/Butcher_link.html, "ESDIRK324L2SA-4-2-3",
/// and PETSc's own src/ts/impls/arkimex/arkimex.c source comment for "ARK-4-2-3" --
/// the task's own "cited to its paper with page/table" is satisfied by the paper
/// itself; these two are the verification that the transcription here matches it,
/// since this round did not fetch the 2003 paper's own PDF directly). Zero diagonal at
/// stage 1 (an EXPLICIT first stage, "ESDIRK"); owner decision (a): a zero diagonal
/// entry is an explicit stage, no new family needed. Stiffly accurate (A's last row ==
/// B, C's last == 1): is_FSAL = true. A- and L-stable, both the method and the order-2
/// embedding (same two sources).
[[nodiscard]] inline ButcherTableau esdirk324l2sa_tableau() {
  ButcherTableau t;
  t.name = "ESDIRK324L2SA";
  t.family = ButcherFamily::DirkEmbedded;
  t.stages = 4;
  t.order = 3;
  t.embedded_order = 2;
  const double gamma = 1767732205903.0 / 4055673282236.0;
  t.A = std::vector<std::vector<double>>{
      {0.0, 0.0, 0.0, 0.0},
      {gamma, gamma, 0.0, 0.0},
      {2746238789719.0 / 10658868560708.0, -640167445237.0 / 6845629431997.0, gamma, 0.0},
      {1471266399579.0 / 7840856788654.0, -4482444167858.0 / 7529755066697.0,
       11266239266428.0 / 11593286722821.0, gamma}
    };
  t.B = std::vector<double>{1471266399579.0 / 7840856788654.0, -4482444167858.0 / 7529755066697.0,
                            11266239266428.0 / 11593286722821.0, gamma};
  t.C = std::vector<double>{0.0, 1767732205903.0 / 2027836641118.0, 3.0 / 5.0, 1.0};
  const std::vector<double> b_hat{2756255671327.0 / 12835298489170.0,
                                  -10771552573575.0 / 22201958757719.0,
                                  9247589265047.0 / 10645013368117.0,
                                  2193209047091.0 / 5459859503100.0};
  std::vector<double> e(4);
  for (std::size_t i = 0; i < 4; ++i) e[i] = t.B[i] - b_hat[i];
  t.E = e;
  t.gamma = std::nullopt;
  t.is_A_stable = true;
  t.is_L_stable = true;
  t.is_S_stable = true;
  t.is_FSAL = true;
  // Round 3 (task item 3): the TM's own "Second-Order Dense Output" table for
  // ARK3(2)4L[2]SA (eq. 49 of the main text, p* = 2), Appendix D, cross-validated
  // against PETSc's own src/ts/impls/arkimex/arkimex.c (`TSARKIMEX3`'s own
  // `binterpt`, which matches the TM's own image byte-for-byte here -- no correction
  // needed, unlike the two higher-order pairs below).
  t.dense_output = std::vector<std::vector<double>>{
      {4655552711362.0 / 22874653954995.0, -215264564351.0 / 13552729205753.0},
      {-18682724506714.0 / 9892148508045.0, 17870216137069.0 / 13817060693119.0},
      {34259539580243.0 / 13192909600954.0, -28141676662227.0 / 17317692491321.0},
      {584795268549.0 / 6622622206610.0, 2508943948391.0 / 7218656332882.0}};
  t.dense_output_order = 2;
  return t;
}

/// SSPRK(3,3) (Shu & Osher, "Efficient implementation of essentially non-oscillatory
/// shock-capturing schemes", J. Comput. Phys. 77 (1988) 439-471; the 3-stage, order-3
/// optimal explicit SSP method, SSP coefficient 1 -- owner decision (b), task item 4's
/// own first-named SSP pair). Shu-Osher form: y1 = y0 + h f(y0); y2 = (3/4) y0 +
/// (1/4) y1 + (1/4) h f(y1); y3 = (1/3) y0 + (2/3) y2 + (2/3) h f(y2) -- converted to
/// Butcher form by the standard substitution (`ssp_coefficient.hpp`'s own
/// `ssp_coefficient_from_shu_osher()` converts it back, task item 4's own "computed,
/// not copied" instruction).
[[nodiscard]] inline ButcherTableau ssprk33_tableau() {
  ButcherTableau t;
  t.name = "SSPRK33";
  t.family = ButcherFamily::Erk;
  t.stages = 3;
  t.order = 3;
  t.embedded_order = std::nullopt;
  t.A = std::vector<std::vector<double>>{
      {0.0, 0.0, 0.0},
      {1.0, 0.0, 0.0},
      {0.25, 0.25, 0.0}
    };
  t.B = std::vector<double>{1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0};
  t.C = std::vector<double>{0.0, 1.0, 0.5};
  t.E = std::nullopt;
  t.gamma = std::nullopt;
  t.is_A_stable = false;
  t.is_L_stable = false;
  t.is_S_stable = false;
  t.is_FSAL = false;
  t.ssp_coefficient = ssp_coefficient_from_shu_osher(ssprk33_shu_osher());
  return t;
}

/// SSPRK(10,4) (Ketcheson, "Highly efficient strong stability-preserving Runge-Kutta
/// methods with low-storage implementations", SIAM J. Sci. Comput. 30 (2008)
/// 2113-2136; the 10-stage, order-4 optimal low-storage explicit SSP method, SSP
/// coefficient 6 -- owner decision (b), task item 4's own second-named SSP pair).
/// Low-storage Shu-Osher recursion (the paper's own Pseudocode 3, reproduced
/// identically by two independent secondary sources: Macdonald, Gottlieb & Ruuth,
/// "On Strong Stability of Explicit Runge-Kutta Methods for Nonlinear Semibounded
/// Operators", and PETSc TS's own TSSSPRKS104): u1 = u0; u_i = u_{i-1} + (h/6) g(u_{i-1})
/// for i = 2..5; u6 = (3/5) u0 + (2/5) (u5 + (h/6) g(u5)); u_i = u_{i-1} + (h/6) g(u_{i-1})
/// for i = 7..10; u_+ = (1/25) u0 + (9/25) (u5 + (h/6) g(u5)) + (3/5) (u10 + (h/6) g(u10)).
/// This factory's own body converts that recursion to the classical (A, B, C) this
/// header's every other tableau already uses, by direct substitution of each u_i back
/// to y0 and the k_j already computed (the ONE "chained Shu-Osher, not single-step"
/// tableau in this file -- `A`'s own rows below are therefore the RESULT of that
/// conversion, not transcribed from any Butcher-form table in the paper, which the
/// paper itself does not give in Butcher form at all; the conversion's own arithmetic
/// is in the factory body's comments, cross-checked by exact Fraction arithmetic
/// before being entered as decimal literals, and proved against the generic
/// order-condition checker: order_conditions_test.cpp's own
/// `RoundTwoTableaus.Ssprk104SatisfiesOrderFourConditions` (written and passing). A
/// SECOND, independent check -- byte-identical agreement against the literal
/// recursion run step by step -- is planned for explicit_integrator_test.cpp but NOT
/// yet written as of this comment; do not cite a specific test name here again until
/// it exists and passes (the prior version of this comment named one that did not).
[[nodiscard]] inline ButcherTableau ssprk104_tableau() {
  ButcherTableau t;
  t.name = "SSPRK104";
  t.family = ButcherFamily::Erk;
  t.stages = 10;
  t.order = 4;
  t.embedded_order = std::nullopt;
  // Expanding the chained recursion in this factory's own doc comment by exact
  // (Fraction) arithmetic -- each u_i's own definition substituted back to y0 and the
  // k_j's already computed, not the single-subdiagonal-entry form a NON-chained
  // Shu-Osher step would give -- produces a LOWER-TRIANGULAR, not sub-diagonal, A:
  // stages 1-4 (u2-u5) each carry h/6 on EVERY earlier stage (because each u_i is
  // itself already the running sum y0 + h6*(k0+...+k_{i-2}), not merely "the previous
  // u plus one new term"); stage 5 (u6) carries (2/5)(h/6) on stages 0-4 uniformly
  // (the (3/5)y0 + (2/5)(...) combination, with the y0 pieces collapsing to exactly 1
  // total weight, verified below); stages 6-9 (u7-u10) add h/6 on each newly-visited
  // stage while CARRYING FORWARD stage 5's own (2/5)(h/6) weight on stages 0-4
  // unchanged (nothing in u7..u10's own "+= h6 g(u_{i-1})" recursion touches those
  // five terms again). B is uniform, 1/10 on every stage -- the final combination's
  // own (1/25, 9/25, 3/5) weights on (y0, stage4-block, stage9-block) collapse to
  // exactly 1/10 on each of the ten k_i once expanded the same way (every stage's own
  // weight comes out identical, a property specific to this OPTIMAL low-storage
  // method, not assumed going in).
  constexpr double h6 = 1.0 / 6.0;
  constexpr double h15 = 1.0 / 15.0;  // (2/5)*h6
  t.A = std::vector<std::vector<double>>{
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {h6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {h6, h6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {h6, h6, h6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {h6, h6, h6, h6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {h15, h15, h15, h15, h15, 0.0, 0.0, 0.0, 0.0, 0.0},
      {h15, h15, h15, h15, h15, h6, 0.0, 0.0, 0.0, 0.0},
      {h15, h15, h15, h15, h15, h6, h6, 0.0, 0.0, 0.0},
      {h15, h15, h15, h15, h15, h6, h6, h6, 0.0, 0.0},
      {h15, h15, h15, h15, h15, h6, h6, h6, h6, 0.0}
    };
  t.B = std::vector<double>{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
  t.C = std::vector<double>{0.0, 1.0 / 6.0, 1.0 / 3.0, 0.5, 2.0 / 3.0,
                            1.0 / 3.0, 0.5, 2.0 / 3.0, 5.0 / 6.0, 1.0};
  // Round 3 (task item 4): Conde, Fekete and Shadid 2018 (arXiv:1806.08693), Section
  // 2.3's own "Embedded pairs for the SSPERK(10,4) method": eight candidate embedded
  // ERK(10,3) rows are given (the propagated SSPRK(10,4) method does not admit a
  // non-defective embedded pair of its own order, per the paper's own section 2.2-2.3
  // argument); the paper's OWN recommendation, b~_3 (chosen for its own stability-
  // radius and error-control measurements, not by this round), is the one shipped
  // here. NOT the classical 3-stage SSPRK(3,3) the task also names: that method has
  // only 3 stages, and this paper's own SSPERK(s,3) embedded-pair family requires
  // s = n^2 (a perfect square, Section 2.2) -- 3 is not one, so no embedded pair for
  // the literal 3-stage method exists in this source; recorded as an open question
  // rather than substituted or fabricated.
  t.embedded_order = 3;
  const std::vector<double> b_hat10{0.0, 2.0 / 9.0, 0.0, 0.0, 5.0 / 18.0, 1.0 / 3.0,
                                    0.0, 0.0, 0.0, 1.0 / 6.0};
  std::vector<double> e10(10);
  for (std::size_t i = 0; i < 10; ++i) e10[i] = t.B[i] - b_hat10[i];
  t.E = e10;
  t.gamma = std::nullopt;
  t.is_A_stable = false;
  t.is_L_stable = false;
  t.is_S_stable = false;
  t.is_FSAL = false;
  t.ssp_coefficient = ssp_coefficient_from_shu_osher(ssprk104_shu_osher());
  return t;
}

/// The Round 2 additions, as a group (parallel to `all_karma_tableaus()`, but for the
/// methods this round added rather than ported).
[[nodiscard]] inline std::vector<ButcherTableau> all_round2_tableaus() {
  return {esdirk324l2sa_tableau(), ssprk33_tableau(), ssprk104_tableau()};
}

/// Kennedy & Carpenter's ESDIRK4(3)6L[2]SA (NASA TM-2001-211038 Appendix D, the
/// ESDIRK half of their ARK4(3)6L[2]SA pair -- the ERK half, used only internally by
/// the additive IMEX pair, is out of this track's own scope, Round 3 task item 2).
/// Coefficients cross-validated against TWO independent machine-readable sources
/// (PETSc src/ts/impls/arkimex/arkimex.c's own `TSARKIMEX4` -- `kennedy_2003` cited
/// in its own doc comment -- and the `qmat` Python package's own
/// `qmat/qcoeff/butcher.py`), after this round's own direct read of the TM's scanned
/// Appendix D image disagreed with both of those on two digits of the dense-output
/// table below (noted at the dense-output assignment itself) -- the two independent,
/// machine-readable sources were trusted over the image read, per the same standard
/// Round 2 held ESDIRK3(2)4L[2]SA's own transcription to.
[[nodiscard]] inline ButcherTableau esdirk436l2sa_tableau() {
  ButcherTableau t;
  t.name = "ESDIRK436L2SA";
  t.family = ButcherFamily::DirkEmbedded;
  t.stages = 6;
  t.order = 4;
  t.embedded_order = 3;
  const double g = 1.0 / 4.0;
  t.A = std::vector<std::vector<double>>{
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {g, g, 0.0, 0.0, 0.0, 0.0},
      {8611.0 / 62500.0, -1743.0 / 31250.0, g, 0.0, 0.0, 0.0},
      {5012029.0 / 34652500.0, -654441.0 / 2922500.0, 174375.0 / 388108.0, g, 0.0, 0.0},
      {15267082809.0 / 155376265600.0, -71443401.0 / 120774400.0, 730878875.0 / 902184768.0,
       2285395.0 / 8070912.0, g, 0.0},
      {82889.0 / 524892.0, 0.0, 15625.0 / 83664.0, 69875.0 / 102672.0, -2260.0 / 8211.0, g}};
  t.B = std::vector<double>{82889.0 / 524892.0, 0.0, 15625.0 / 83664.0, 69875.0 / 102672.0,
                            -2260.0 / 8211.0, g};
  t.C = std::vector<double>{0.0, 1.0 / 2.0, 83.0 / 250.0, 31.0 / 50.0, 17.0 / 20.0, 1.0};
  const std::vector<double> b_hat{4586570599.0 / 29645900160.0, 0.0, 178811875.0 / 945068544.0,
                                  814220225.0 / 1159782912.0, -3700637.0 / 11593932.0,
                                  61727.0 / 225920.0};
  std::vector<double> e(6);
  for (std::size_t i = 0; i < 6; ++i) e[i] = t.B[i] - b_hat[i];
  t.E = e;
  t.gamma = std::nullopt;
  t.is_A_stable = true;   // L-stable per the TM's own text (Section 5, "L stable, stage order 2")
  t.is_L_stable = true;
  t.is_S_stable = false;  // not computed this round; not claimed either way in the TM's own text
  t.is_FSAL = true;       // Structural property: A's last row equals B, c_s = 1 -- genuinely
                          // stiffly accurate. Round 3, item 6: DirkIntegrator's own FSAL reuse
                          // mechanism exists and is tested (dirk_integrator_test.cpp's own FSAL
                          // suite) but ships OPT-IN, not triggered by this flag alone -- MEASURED
                          // not bit-identical to full recomputation for the implicit family (see
                          // DirkIntegrator::set_fsal_enabled()'s own comment for the finding).
  // Round 3 (task item 2a): the TM's own "Third-Order Dense Output" table (eq. 54, p* = 3),
  // Appendix D. Two digits corrected against the TM's own image read, using PETSc's `binterpt`
  // and `qmat`'s cross-agreeing values: stage 1's third-column numerator
  // (6818779370841 -> 6818779379841) and stage 3's third-column numerator
  // (2173542090792 -> 2173542590792).
  t.dense_output = std::vector<std::vector<double>>{
      {6943876665148.0 / 7220017795957.0, -54480133.0 / 30881146.0, 6818779379841.0 / 7100303317025.0},
      {0.0, 0.0, 0.0},
      {7640104374378.0 / 9702883013639.0, -11436875.0 / 14766696.0, 2173542590792.0 / 12501825683035.0},
      {-20649996744609.0 / 7521556579894.0, 174696575.0 / 18121608.0, -31592104683404.0 / 5083833661969.0},
      {8854892464581.0 / 2390941311638.0, -12120380.0 / 966161.0, 61146701046299.0 / 7138195549469.0},
      {-11397109935349.0 / 6675773540249.0, 3843.0 / 706.0, -17219254887155.0 / 4939391667607.0}};
  t.dense_output_order = 3;
  return t;
}

/// ESDIRK5(4)8L[2]SA: the ESDIRK half of ARK5(4)8L[2]SA, Kennedy and Carpenter, NASA
/// TM-2001-211038 (Appl. Numer. Math. 44 (2003) 139-181), Appendix D, read from the
/// scanned page images; cross-checked against PETSc's TSARKIMEX5 and qmat's
/// ARK548L2SAESDIRK (NOT ARK548L2SAESDIRK2, a different 2019 tableau) where the scan's
/// text layer showed OCR damage. Order 5, embedded order 4, stiffly accurate,
/// L-stable, stage order 2, gamma = 41/200, dense output of order 3 (eq. 54's
/// b_i(theta) rows, verified by the continuous order conditions).
///
/// Order, verified exactly (lead, 2026-10-31, fractions.Fraction on the shipped
/// rationals): every rooted-tree condition through order 5 holds with a residual of
/// at most 5.3e-27 -- the same magnitude as orders 1 to 4 (1.1e-26 to 5.3e-27), the
/// precision of the published 13-digit rational approximations of the true
/// coefficients -- and order 6 fails by 4.2e-3; the embedded weights hold through
/// order 4 at 7.3e-27 and fail order 5 by 1.3e-4. The round-3 track had declared
/// this tableau order 4 with embedded order 3 after reading "the order-5 conditions
/// also pass" as an unresolvable 4-versus-5 distinction; it is order 5, as its name
/// says, and the measured orders agree (logistic 4.991 +/- 0.006, non-autonomous
/// 5.004 +/- 0.001, Dahlquist 4.97 +/- 0.04 on the truncation-limited range).
[[nodiscard]] inline ButcherTableau esdirk548l2sa_tableau() {
  ButcherTableau t;
  t.name = "ESDIRK548L2SA";
  t.family = ButcherFamily::DirkEmbedded;
  t.stages = 8;
  t.order = 5;
  t.embedded_order = 4;
  const double g = 41.0 / 200.0;
  t.A = std::vector<std::vector<double>>{
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {g, g, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {41.0 / 400.0, -567603406766.0 / 11931857230679.0, g, 0.0, 0.0, 0.0, 0.0, 0.0},
      {683785636431.0 / 9252920307686.0, 0.0, -110385047103.0 / 1367015193373.0, g, 0.0, 0.0, 0.0, 0.0},
      {3016520224154.0 / 10081342136671.0, 0.0, 30586259806659.0 / 12414158314087.0,
       -22760509404356.0 / 11113319521817.0, g, 0.0, 0.0, 0.0},
      {218866479029.0 / 1489978393911.0, 0.0, 638256894668.0 / 5436446318841.0,
       -1179710474555.0 / 5321154724896.0, -60928119172.0 / 8023461067671.0, g, 0.0, 0.0},
      {1020004230633.0 / 5715676835656.0, 0.0, 25762820946817.0 / 25263940353407.0,
       -2161375909145.0 / 9755907335909.0, -211217309593.0 / 5846859502534.0,
       -4269925059573.0 / 7827059040749.0, g, 0.0},
      {-872700587467.0 / 9133579230613.0, 0.0, 0.0, 22348218063261.0 / 9555858737531.0,
       -1143369518992.0 / 8141816002931.0, -39379526789629.0 / 19018526304540.0,
       32727382324388.0 / 42900044865799.0, g}};
  t.B = std::vector<double>{-872700587467.0 / 9133579230613.0, 0.0, 0.0,
                            22348218063261.0 / 9555858737531.0, -1143369518992.0 / 8141816002931.0,
                            -39379526789629.0 / 19018526304540.0, 32727382324388.0 / 42900044865799.0, g};
  t.C = std::vector<double>{0.0, 41.0 / 100.0, 2935347310677.0 / 11292855782101.0,
                            1426016391358.0 / 7196633302097.0, 92.0 / 100.0, 24.0 / 100.0,
                            3.0 / 5.0, 1.0};
  const std::vector<double> b_hat{-975461918565.0 / 9796059967033.0, 0.0, 0.0,
                                  78070527104295.0 / 32432590147079.0, -548382580838.0 / 3424219808633.0,
                                  -33438840321285.0 / 15594753105479.0, 3629800801594.0 / 4656183773603.0,
                                  4035322873751.0 / 18575991585200.0};
  std::vector<double> e(8);
  for (std::size_t i = 0; i < 8; ++i) e[i] = t.B[i] - b_hat[i];
  t.E = e;
  t.gamma = std::nullopt;
  t.is_A_stable = true;
  t.is_L_stable = true;
  t.is_S_stable = false;
  t.is_FSAL = true;       // Structural property (same basis as esdirk436l2sa_tableau()'s own
                          // comment above): stiffly accurate, A's last row equals B, c_s = 1.
                          // DirkIntegrator's own FSAL reuse ships opt-in, not default, for the
                          // same measured reason.
  // Round 3 (task item 2b): the TM's own "Third-Order Dense Output" table (p* = 3),
  // Appendix D. Several digits corrected against the TM's own image read using PETSc's
  // `binterpt` (this round's own image read of this specific 8-stage table proved
  // UNRELIABLE across several entries, confirming the 2-independent-source bar's own
  // worth): stage 4 (numerator/denominator digits in columns 1-2), stage 5 (digits in
  // all three columns), stage 7 (column-2 denominator and column-3 numerator).
  t.dense_output = std::vector<std::vector<double>>{
      {-17674230611817.0 / 10670229744614.0, 43486358583215.0 / 12773830924787.0,
       -9257016797708.0 / 5021505065439.0},
      {0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0},
      {65168852399939.0 / 7868540260826.0, -91478233927265.0 / 11067650958493.0,
       26096422576131.0 / 11239449250142.0},
      {15494834004392.0 / 5936557850923.0, -79368583304911.0 / 10890268929626.0,
       92396832856987.0 / 20362823103730.0},
      {-99329723586156.0 / 26959484932159.0, -12239297817655.0 / 9152339842473.0,
       30029262896817.0 / 10175596800299.0},
      {-19024464361622.0 / 5461577185407.0, 115839755401235.0 / 10719374521269.0,
       -26136350496073.0 / 3983972220547.0},
      {-6511271360970.0 / 6095937251113.0, 5843115559534.0 / 2180450260947.0,
       -5289405421727.0 / 3760307252460.0}};
  t.dense_output_order = 3;
  return t;
}

/// The Round 3 additions, as a group (parallel to `all_round2_tableaus()`): the two
/// new ESDIRK pairs. SSPRK33/SSPRK104 are unchanged tableaus (their own Round 2
/// factory functions now also carry a dense-output or embedded row, but are still the
/// SAME ButcherTableau objects, not duplicated here).
[[nodiscard]] inline std::vector<ButcherTableau> all_round3_tableaus() {
  return {esdirk436l2sa_tableau(), esdirk548l2sa_tableau()};
}

/// Every factory above, for a test or a report that wants "all 21" without naming each
/// one. Order matches the RKType enum's own declaration order
/// (include/enums/runge_kutta_schemes.h:42-64).
[[nodiscard]] inline std::vector<ButcherTableau> all_karma_tableaus() {
  return {
    forward_euler_tableau(), kutta22_tableau(), kutta44_tableau(), dopri45_tableau(),
    backward_euler_tableau(), crank_nicolson_tableau(),
    sdirk22_tableau(), sdirk23_tableau(), sdirk34_tableau(), sdirk332_tableau(),
    sdirk543_tableau(), sdirk643_tableau(), trbe_tableau(), sdirk33_tableau(),
    ros22_tableau(), grk4a_tableau(), grk4t_tableau(),
    grk4sn_tableau(), ros3p_tableau(), rodasp_tableau(), sdirk44_tableau(),
  };
}

}  // namespace fuse::num
