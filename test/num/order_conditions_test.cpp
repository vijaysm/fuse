// S8 physics-kit, track "Integrators", Stage B Round 2, task item 1.
//
// A generic order-condition checker (fuse/num/order_conditions.hpp) over Butcher's own
// rooted trees, proved here against known published per-tree conditions before being
// trusted to grade a single tableau (the task's own instruction, "do not hand-check" --
// this file IS the hand-check, run once, of the generic machinery itself).

#include "fuse/num/order_conditions.hpp"

#include "fuse/base/error.hpp"

#include <cmath>
#include <limits>
#include <vector>

#include <stdexcept>

#include <gtest/gtest.h>

namespace fuse::num {
namespace {

// Same derivation style as butcher_tableau_test.cpp's own consistency_tolerance: the
// order-condition evaluation chains at most 5 nested products/sums of at most `stages`
// terms each (order-5 trees have at most 4 children, each contributing one
// `stages`-term inner sum) -- 1024 * stages^2 * eps is generous against that, MEASURED
// (with this bound removed) to be at least 3 orders of magnitude above every shipped
// classical tableau's own observed defect through its declared order (worst observed
// 1.11e-15, SDIRK543/SDIRK643, 6 stages) and at least 2 orders of magnitude below every
// genuine order-condition failure this file or butcher_tableau.hpp's own top comment
// measures (smallest genuine failure among the existing tableaus: KUTTA44's own
// next-order defect, 8.3e-3; SDIRK44's own finding, exactly 1/18).
double order_tolerance(int stages) {
  return 1024.0 * static_cast<double>(stages) * static_cast<double>(stages) *
         std::numeric_limits<double>::epsilon();
}

bool is_classical_family(ButcherFamily f) {
  switch (f) {
    case ButcherFamily::Erk:
    case ButcherFamily::ErkEmbedded:
    case ButcherFamily::Dirk:
    case ButcherFamily::DirkEmbedded:
    case ButcherFamily::Sdirk:
    case ButcherFamily::SdirkEmbedded:
    case ButcherFamily::FirkNonSingular:
    case ButcherFamily::FirkSingular:
      return true;
    case ButcherFamily::Rosenbrock:
    case ButcherFamily::RosenbrockEmbedded:
      return false;
  }
  return false;
}

constexpr int kMaxCatalogOrder = 6;  // 6 since 2026-10-31: ESDIRK548L2SA is order 5 and its order-6 defect must be checked

}  // namespace

// OEIS A000081 (rooted trees by node count): 1, 1, 2, 4, 9, 20 for n = 1..6. A generator
// that is missing a tree, or double-generating one as two "different" catalog entries,
// is caught here before it ever grades a single tableau.
TEST(OrderConditions, CatalogSizesMatchOeisA000081) {
  RootedTreeCatalog cat(kMaxCatalogOrder);
  const std::vector<std::size_t> expected{1, 1, 2, 4, 9, 20};
  for (int n = 1; n <= kMaxCatalogOrder; ++n) {
    EXPECT_EQ(cat.of_order(n).size(), expected[static_cast<std::size_t>(n - 1)])
        << "order " << n;
  }
  std::size_t total = 0;
  for (int n = 1; n <= kMaxCatalogOrder; ++n) total += cat.of_order(n).size();
  EXPECT_EQ(total, 37u);
  EXPECT_EQ(cat.trees().size(), 37u);
}

// An order the catalog was not built to is refused, not read past the end of the
// per-order table (lead, 2026-10-31: on a catalog built to 5, of_order(6) returned an
// empty range through out-of-bounds indexing, so a "next order must fail" check
// passed with a defect of exactly 0).
TEST(OrderConditions, AnOrderBeyondTheCatalogIsRefused) {
  RootedTreeCatalog cat(3);
  EXPECT_EQ(cat.of_order(3).size(), 2u);
  EXPECT_THROW((void)cat.of_order(4), fuse::InvalidArgument);
  EXPECT_THROW((void)cat.of_order(0), fuse::InvalidArgument);
  EXPECT_THROW(max_defect_at_order(cat, kutta44_tableau().A, kutta44_tableau().B, 4), fuse::InvalidArgument);
}

// Butcher's density for the first eight trees (orders 1-4), pinned against the
// published table (Hairer, Norsett & Wanner, Solving ODEs I, Table II.2.1): 1, 2, 3, 6,
// 4, 8, 12, 24 -- independent of any tableau, gamma(t) depends only on the tree.
TEST(OrderConditions, DensityMatchesThePublishedTable) {
  RootedTreeCatalog cat(kMaxCatalogOrder);
  std::vector<double> gammas;
  for (int ord = 1; ord <= 4; ++ord) {
    for (int tid : cat.of_order(ord)) gammas.push_back(cat.trees()[static_cast<std::size_t>(tid)].gamma);
  }
  std::sort(gammas.begin(), gammas.end());
  const std::vector<double> expected{1, 2, 3, 4, 6, 8, 12, 24};
  ASSERT_EQ(gammas.size(), expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) EXPECT_DOUBLE_EQ(gammas[i], expected[i]);
}

// The catalog's own Phi recursion, proved against four HAND-derived literature
// formulas (Hairer, Norsett & Wanner, Solving ODEs I, Section II.2's own order-4
// condition list: b.c^3=1/4, sum b_i c_i (Ac)_i=1/8, sum b_i (Ac^2)_i=1/12,
// sum b_i (A(Ac))_i=1/24) on the classical 4-stage RK4 tableau, computed TWO
// independent ways and checked to agree -- this is what proves `stage_weights()`'s own
// per-child-independent-product recursion (its own doc comment explains why a
// single shared sum across children is the WRONG formula) is the right one, not merely
// self-consistent.
TEST(OrderConditions, PerChildProductMatchesTheLiteratureConditions) {
  const ButcherTableau t = kutta44_tableau();
  RootedTreeCatalog cat(kMaxCatalogOrder);
  const auto phi = stage_weights(cat, t.A);

  // Hand formulas, directly from A/B/C (no tree machinery).
  const auto& A = t.A;
  const auto& B = t.B;
  const auto& C = t.C;
  const std::size_t s = B.size();
  double hand_c3 = 0.0, hand_cAc = 0.0, hand_Ac2 = 0.0, hand_AAc = 0.0;
  std::vector<double> Ac(s, 0.0), Ac2(s, 0.0), AAc(s, 0.0);
  for (std::size_t i = 0; i < s; ++i) {
    for (std::size_t j = 0; j < s; ++j) Ac[i] += A[i][j] * C[j];
  }
  for (std::size_t i = 0; i < s; ++i) {
    for (std::size_t j = 0; j < s; ++j) Ac2[i] += A[i][j] * C[j] * C[j];
  }
  for (std::size_t i = 0; i < s; ++i) {
    for (std::size_t j = 0; j < s; ++j) AAc[i] += A[i][j] * Ac[j];
  }
  for (std::size_t i = 0; i < s; ++i) {
    hand_c3 += B[i] * C[i] * C[i] * C[i];
    hand_cAc += B[i] * C[i] * Ac[i];
    hand_Ac2 += B[i] * Ac2[i];
    hand_AAc += B[i] * AAc[i];
  }

  // Tree-driven: the four order-4 trees are, by children multiset (see
  // order_conditions.hpp's own construction, catalog ids assigned in generation
  // order): three-leaf star [0,0,0], one-leaf-plus-order2-chain [0,1], one order-3
  // "bush" child [2], one order-3 "chain" child [3] -- identified here by sorted
  // children rather than hand-picked ids, so this test does not depend on the
  // catalog's own id-assignment order surviving unchanged.
  double tree_c3 = 0.0, tree_cAc = 0.0, tree_Ac2 = 0.0, tree_AAc = 0.0;
  for (int tid : cat.of_order(4)) {
    const RootedTree& tr = cat.trees()[static_cast<std::size_t>(tid)];
    double val = 0.0;
    for (std::size_t i = 0; i < s; ++i) val += B[i] * phi[static_cast<std::size_t>(tid)][i];
    if (tr.children.size() == 3) {
      tree_c3 = val;
    } else if (tr.children.size() == 2) {
      tree_cAc = val;
    } else {
      // single child: distinguish the order-3 "bush" grandchild (gamma 3) from the
      // order-3 "chain" grandchild (gamma 6) by the CHILD's own gamma, not its id.
      const double child_gamma = cat.trees()[static_cast<std::size_t>(tr.children[0])].gamma;
      if (child_gamma == 3.0) {
        tree_Ac2 = val;
      } else {
        tree_AAc = val;
      }
    }
  }
  // Derived (lead, 2026-10-31; round 2 had a chosen 1e-13): the two evaluations
  // of one elementary weight differ only in operation order, each being at most
  // s^3 + s products-and-sums of operands of magnitude at most 1 (|A|, |B|, |C| <= 1
  // for Kutta44), so the difference is at most 2 * (s^3 + s) * eps * (sum of the
  // magnitudes of the terms), and the summed magnitudes are each at most 1 here
  // (the weights sum to 1 and every factor is in [0, 1]).
  const double kTol = 2.0 * (static_cast<double>(s * s * s + s)) * std::numeric_limits<double>::epsilon();
  EXPECT_NEAR(tree_c3, hand_c3, kTol);
  EXPECT_NEAR(tree_cAc, hand_cAc, kTol);
  EXPECT_NEAR(tree_Ac2, hand_Ac2, kTol);
  EXPECT_NEAR(tree_AAc, hand_AAc, kTol);
  // And both agree with the published targets (RK4 is genuinely order 4).
  EXPECT_NEAR(hand_c3, 1.0 / 4.0, kTol);
  EXPECT_NEAR(hand_cAc, 1.0 / 8.0, kTol);
  EXPECT_NEAR(hand_Ac2, 1.0 / 12.0, kTol);
  EXPECT_NEAR(hand_AAc, 1.0 / 24.0, kTol);
}

// Every classical-family tableau (Erk/Dirk/Sdirk/Firk*) in all_karma_tableaus()
// satisfies every order condition through its own declared `order` to round-off, AND
// does not understate it: at least one condition at `order + 1` fails by well above
// round-off (skipped when `order + 1` exceeds the catalog's own max order, 5 --
// DOPRI45 is order 5 already). This is the task's own "prove every new tableau" in its
// general form, run here against the 15 EXISTING classical tableaus first.
TEST(OrderConditions, ClassicalFamiliesSatisfyExactlyTheirDeclaredOrder) {
  RootedTreeCatalog cat(kMaxCatalogOrder);
  for (const ButcherTableau& t : all_karma_tableaus()) {
    if (!is_classical_family(t.family)) continue;
    const double tol = order_tolerance(t.stages);
    const double thru = max_defect_through_order(cat, t.A, t.B, std::min(t.order, kMaxCatalogOrder));
    EXPECT_LT(thru, tol) << t.name << ": defect through order " << t.order << " = " << thru;
    if (t.order < kMaxCatalogOrder) {
      const double next = max_defect_at_order(cat, t.A, t.B, t.order + 1);
      EXPECT_GT(next, 1000.0 * tol)
          << t.name << ": order " << (t.order + 1) << " condition should NOT hold "
          << "(method is declared order " << t.order << " exactly), defect = " << next;
    }
    if (t.has_embedded_estimate()) {
      const auto b_hat = embedded_weights(t);
      const double ethru = max_defect_through_order(
          cat, t.A, b_hat, std::min(*t.embedded_order, kMaxCatalogOrder));
      EXPECT_LT(ethru, tol) << t.name << ": embedded defect through order "
                            << *t.embedded_order << " = " << ethru;
    }
  }
}

// SDIRK44's own finding (butcher_tableau.hpp's own top comment, fifth finding): every
// condition through order 3 holds to round-off, but the order-4 condition
// `sum_i b_i c_i^3 = 1/4` fails by EXACTLY 1/18 -- pinned as an exact number (not just
// "fails") so a future accidental "fix" of the coefficients that happened to also
// satisfy b.c^3=1/4 would be caught by this test actually changing value, prompting a
// second look, rather than the test simply going on passing by the metadata already
// having been corrected to order 3.
TEST(OrderConditions, Sdirk44sDeclaredOrderIsCorrectedFromFourToThree) {
  const ButcherTableau t = sdirk44_tableau();
  EXPECT_EQ(t.order, 3);
  RootedTreeCatalog cat(kMaxCatalogOrder);
  const double thru3 = max_defect_through_order(cat, t.A, t.B, 3);
  EXPECT_NEAR(thru3, 0.0, order_tolerance(t.stages));
  const double at4 = max_defect_at_order(cat, t.A, t.B, 4);
  EXPECT_NEAR(at4, 1.0 / 18.0, 1e-12)
      << "sdirk44.C's own order-4 defect must stay exactly 1/18, not merely nonzero, so "
         "a coefficient edit that happens to also satisfy b.c^3=1/4 is still caught";
}

// The Rosenbrock family's own scope limit (butcher_tableau.hpp's own top comment, fifth
// finding): `rosenbrock_classical_form()`'s (M, b) satisfies the classical order-2
// conditions for EVERY Rosenbrock tableau (the Jacobian-exactness equivalence holds
// that far unconditionally), but the higher-order classical conditions do NOT hold for
// any tableau declared order 3 or above -- this is not a defect, it is why task item 3
// asks for a MEASURED numerical order test on the Rosenbrock family rather than this
// checker; the test exists so a future change that silently started asserting the
// higher-order classical conditions for Rosenbrock (which would be mathematically
// unjustified, Hairer & Wanner Solving ODEs II IV.7) is caught by this test FAILING to
// fail, not by a missing assertion nobody noticed.
TEST(OrderConditions, RosenbrockOrderTwoConditionsHoldButHigherOrderClassicalConditionsDoNot) {
  RootedTreeCatalog cat(kMaxCatalogOrder);
  for (const ButcherTableau& t : all_karma_tableaus()) {
    if (is_classical_family(t.family)) continue;
    const ClassicalForm cf = rosenbrock_classical_form(t);
    const double tol = order_tolerance(t.stages);
    const double thru2 = max_defect_through_order(cat, cf.M, cf.b, 2);
    EXPECT_LT(thru2, tol) << t.name << ": classical order-2 defect = " << thru2;
    if (t.order >= 3) {
      const double at_declared = max_defect_at_order(cat, cf.M, cf.b, t.order);
      EXPECT_GT(at_declared, 1000.0 * tol)
          << t.name << ": classical order-" << t.order << " condition should NOT hold "
          << "for a Rosenbrock tableau (Jacobian-exactness equivalence stops at order 2)";
    }
  }
}

// Round 2's own new tableaus (task items 2 and 4): each proved by this SAME generic
// checker before being wired into any integrator, per the task's own "every tableau
// you add must pass it... before it is used anywhere".
TEST(RoundTwoTableaus, EveryNewTableauSatisfiesExactlyItsDeclaredOrder) {
  RootedTreeCatalog cat(kMaxCatalogOrder);
  for (const ButcherTableau& t : all_round2_tableaus()) {
    const double tol = order_tolerance(t.stages);
    const double thru = max_defect_through_order(cat, t.A, t.B, t.order);
    EXPECT_LT(thru, tol) << t.name << ": defect through order " << t.order << " = " << thru;
    const double next = max_defect_at_order(cat, t.A, t.B, t.order + 1);
    EXPECT_GT(next, 1000.0 * tol)
        << t.name << ": order " << (t.order + 1) << " should NOT hold, defect = " << next;
    if (t.has_embedded_estimate()) {
      const auto b_hat = embedded_weights(t);
      const double ethru =
          max_defect_through_order(cat, t.A, b_hat, *t.embedded_order);
      EXPECT_LT(ethru, tol) << t.name << ": embedded defect through order "
                            << *t.embedded_order << " = " << ethru;
    }
  }
}

// SSPRK(10,4)'s own order-4 proof by name, since this is the one new tableau whose
// A/B were derived by hand (Shu-Osher substitution, butcher_tableau.hpp's own
// ssprk104_tableau() doc comment) rather than copied from a published Butcher table --
// a defect here would mean the substitution itself was wrong, not a transcription slip.
TEST(RoundTwoTableaus, Ssprk104SatisfiesOrderFourConditions) {
  const ButcherTableau t = ssprk104_tableau();
  RootedTreeCatalog cat(kMaxCatalogOrder);
  const double thru4 = max_defect_through_order(cat, t.A, t.B, 4);
  EXPECT_LT(thru4, order_tolerance(t.stages)) << "defect through order 4 = " << thru4;
  const double at5 = max_defect_at_order(cat, t.A, t.B, 5);
  EXPECT_GT(at5, 1e-6) << "order 5 should NOT hold for a declared order-4 method";
}

// S8 physics-kit, track "Integrators", Stage B Round 3 (task items 1-3): the two new
// Kennedy & Carpenter ESDIRK pairs, proved the SAME way as every tableau this project
// ships -- row sums, order conditions through the declared order, embedded weights to
// embedded order, stiff accuracy, and (new this round) the dense-output table's own
// b_i(1)=b_i reproduction. "A tableau that fails any of these is NOT shipped" (the
// task's own instruction). Both pairs resolve cleanly: ESDIRK436L2SA holds through
// order 4 exactly and fails order 5 by 9.2e-3; ESDIRK548L2SA holds through order 5
// at the 1e-27 precision of its published rationals and fails order 6 by 4.2e-3
// (lead, 2026-10-31; the round-3 track had declared it order 4 and this test
// "unresolvable" for it after misreading the passing order-5 conditions).
TEST(RoundThreeTableaus, EsdirkPairsPassEveryCheckTheAlgebraicCheckerCanResolve) {
  RootedTreeCatalog cat(kMaxCatalogOrder);
  struct Case { const char* name; ButcherTableau t; bool order_distinction_resolvable; };
  std::vector<Case> cases;
  cases.push_back({"ESDIRK436L2SA", esdirk436l2sa_tableau(), true});
  cases.push_back({"ESDIRK548L2SA", esdirk548l2sa_tableau(), true});
  for (const auto& c : cases) {
    const ButcherTableau& t = c.t;
    const double tol = order_tolerance(t.stages);
    // Row sums (consistency): max_i |C[i] - sum_j A[i][j]|.
    double row_defect = 0.0;
    for (int i = 0; i < t.stages; ++i) {
      double s = 0.0;
      for (int j = 0; j < t.stages; ++j) s += t.A[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
      row_defect = std::max(row_defect, std::fabs(s - t.C[static_cast<std::size_t>(i)]));
    }
    EXPECT_LT(row_defect, tol) << c.name << ": row-sum defect = " << row_defect;
    // Stiff accuracy: last row of A equals B.
    EXPECT_EQ(t.A.back(), t.B) << c.name << ": last row of A must equal B (stiffly accurate)";
    // Order through the declared order, and the next order must FAIL.
    const double thru = max_defect_through_order(cat, t.A, t.B, t.order);
    EXPECT_LT(thru, tol) << c.name << ": defect through order " << t.order << " = " << thru;
    if (c.order_distinction_resolvable) {
      const double next = max_defect_at_order(cat, t.A, t.B, t.order + 1);
      EXPECT_GT(next, 1000.0 * tol)
          << c.name << ": order " << (t.order + 1) << " should NOT hold, defect = " << next;
    }
    // Embedded order.
    ASSERT_TRUE(t.has_embedded_estimate()) << c.name << ": must carry an embedded estimate";
    const auto b_hat = embedded_weights(t);
    const double ethru = max_defect_through_order(cat, t.A, b_hat, *t.embedded_order);
    EXPECT_LT(ethru, tol) << c.name << ": embedded defect through order " << *t.embedded_order
                          << " = " << ethru;
    // Dense-output b_i(1) = b_i (eq. 54's own row-sum property).
    ASSERT_TRUE(t.has_dense_output()) << c.name << ": must carry a dense-output table";
    double dense_defect = 0.0;
    for (std::size_t i = 0; i < t.dense_output->size(); ++i) {
      double s = 0.0;
      for (double coeff : (*t.dense_output)[i]) s += coeff;
      dense_defect = std::max(dense_defect, std::fabs(s - t.B[i]));
    }
    // Derived bound: each row sums k coefficients already rounded to double, so the
    // computed sum differs from the exact one by at most (k + 1) * eps * sum|coeff|
    // (Higham, recursive summation, plus the rounding of B[i] itself); the published
    // rationals' own 1e-27 residual is far below that. The round-3 track had used a
    // chosen 1e-13 here (lead, 2026-10-31).
    double dense_bound = 0.0;
    for (std::size_t i = 0; i < t.dense_output->size(); ++i) {
      double abs_sum = std::fabs(t.B[i]);
      for (double coeff : (*t.dense_output)[i]) abs_sum += std::fabs(coeff);
      dense_bound = std::max(dense_bound, (static_cast<double>((*t.dense_output)[i].size()) + 1.0) *
                                              std::numeric_limits<double>::epsilon() * abs_sum);
    }
    EXPECT_LT(dense_defect, dense_bound)
        << c.name << ": dense-output b_i(1)=b_i defect = " << dense_defect << " (bound " << dense_bound << ")";
  }
}

// The ARK3(2)4L[2]SA dense-output addition (task item 3) to the ALREADY-shipped
// ESDIRK324L2SA tableau -- same b_i(1)=b_i check, same tolerance reasoning.
TEST(RoundThreeTableaus, Esdirk324l2saDenseOutputReproducesB) {
  const ButcherTableau t = esdirk324l2sa_tableau();
  ASSERT_TRUE(t.has_dense_output());
  double worst = 0.0, bound = 0.0;
  for (std::size_t i = 0; i < t.dense_output->size(); ++i) {
    double s = 0.0, abs_sum = std::fabs(t.B[i]);
    for (double coeff : (*t.dense_output)[i]) { s += coeff; abs_sum += std::fabs(coeff); }
    worst = std::max(worst, std::fabs(s - t.B[i]));
    // Same derived bound as EsdirkPairsPassEveryCheckTheAlgebraicCheckerCanResolve.
    bound = std::max(bound, (static_cast<double>((*t.dense_output)[i].size()) + 1.0) *
                                std::numeric_limits<double>::epsilon() * abs_sum);
  }
  EXPECT_LT(worst, bound) << "ESDIRK324L2SA dense-output b_i(1)=b_i defect = " << worst << " (bound " << bound << ")";
}

// SSPRK(10,4)'s own embedded pair (task item 4, Conde/Fekete/Shadid 2018) -- order 3,
// algebraically distinct from the propagated order 4.
TEST(RoundThreeTableaus, Ssprk104EmbeddedPairIsOrderThree) {
  const ButcherTableau t = ssprk104_tableau();
  ASSERT_TRUE(t.has_embedded_estimate());
  RootedTreeCatalog cat(kMaxCatalogOrder);
  const auto b_hat = embedded_weights(t);
  const double thru3 = max_defect_through_order(cat, t.A, b_hat, 3);
  EXPECT_LT(thru3, order_tolerance(t.stages)) << "embedded defect through order 3 = " << thru3;
  const double at4 = max_defect_at_order(cat, t.A, b_hat, 4);
  EXPECT_GT(at4, 1e-6) << "the embedded pair's own order 4 should NOT hold (it is an "
                          "order-3 ERROR ESTIMATE, not a second order-4 method)";
}

}  // namespace fuse::num
