#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

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

  /// The honest query Section 4.2 of the design doc (EmbeddedController) and
  /// embedded_controller.hpp's own refusals are built on: true only when both an
  /// embedded order AND a (not-necessarily-nonzero, but AT LEAST present) `E` are on
  /// record. GRK4A therefore reports false here, by construction, not by a caller
  /// remembering to special-case it.
  [[nodiscard]] bool has_embedded_estimate() const noexcept {
    return embedded_order.has_value() && E.has_value();
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


/// SDIRK44 (sdirk44.C:22-39).
[[nodiscard]] inline ButcherTableau sdirk44_tableau() {
  ButcherTableau t;
  t.name = "SDIRK44";
  t.family = ButcherFamily::Sdirk;
  t.stages = 4;
  t.order = 4;
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
